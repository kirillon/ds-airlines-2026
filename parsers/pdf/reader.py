"""Bounded, sequential extraction of two-column SkyTeam timetable PDFs."""

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pypdfium2 as pdfium

from models.contracts.schedules import ScheduleRecord

_MONTHS = {
    name: i
    for i, name in enumerate(
        "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1
    )
}
_COVER = re.compile(
    r"Covers period:\s*(\d{2}) (\w{3}) (\d{4}) through (\d{2}) (\w{3}) (\d{4})"
)
_ROUTE = re.compile(r"^(FROM|TO):.*\b([A-Z]{3})\s*$")
_ROW = re.compile(
    r"^(\d{2}) (\w{3})\s*-\s*(\d{2}) (\w{3})\s*([1-7 ]+?)\s+"
    r"(\d{2}:\d{2})\s+(\d{2}:\d{2})(?:\s*((?:\+-|[+-])\d))?\s+"
    r"([A-Z0-9]{2}\d{1,4}[A-Z]?)(\*)?\s+(\S+)\s+(?:(\d+)H)?(?:(\d+)M)?$"
)
_HEADERS = {"Time", "Arr", "Flight Aircraft Travel", "Validity Days Dep"}


_LOGGER = logging.getLogger(__name__)


class PdfParseError(ValueError):
    """The input cannot be read as a supported timetable."""


class ResourceLimitError(PdfParseError):
    """The input exceeds a configured processing limit."""


class UnsupportedLayoutError(PdfParseError):
    """The document does not follow the supported table layout."""


@dataclass(frozen=True, slots=True)
class ParseOptions:
    batch_size: int = 5000
    max_pages: int = 100000
    max_file_bytes: int = 1024**3
    max_page_chars: int = 200000
    valid_from: date | None = None
    valid_to: date | None = None

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value <= 0
            for value in (
                self.batch_size,
                self.max_pages,
                self.max_file_bytes,
                self.max_page_chars,
            )
        ):
            raise ValueError("resource limits must be positive integers")
        if (self.valid_from is None) != (self.valid_to is None):
            raise ValueError("both validity bounds are required")
        if (
            self.valid_from is not None
            and self.valid_to is not None
            and self.valid_from > self.valid_to
        ):
            raise ValueError("validity bounds must be ordered")


def _resolve_dates(
    start_day: str,
    start_month: str,
    end_day: str,
    end_month: str,
    bounds: tuple[date, date] | None,
) -> tuple[date, date]:
    if bounds is None:
        raise ValueError("missing validity bounds")
    lower, upper = bounds
    candidates = []
    for year in range(lower.year, upper.year + 1):
        try:
            start = date(year, _MONTHS[start_month], int(start_day))
            end_year = year + (_MONTHS[end_month] < _MONTHS[start_month])
            end = date(end_year, _MONTHS[end_month], int(end_day))
        except (KeyError, ValueError):
            continue
        if lower <= start <= end <= upper:
            candidates.append((start, end))
    if len(candidates) != 1:
        raise ValueError("ambiguous or invalid yearless dates")
    return candidates[0]


def _parse_clock(value: str) -> int:
    hour, minute = map(int, value.split(":"))
    if hour > 23 or minute > 59:
        raise ValueError("invalid local time")
    return hour * 60 + minute


def _build_record(
    match: re.Match[str],
    context: list[str | None],
    bounds: tuple[date, date] | None,
) -> ScheduleRecord:
    (
        sd,
        sm,
        ed,
        em,
        days,
        departure,
        arrival,
        offset,
        flight,
        star,
        aircraft,
        hours,
        minutes,
    ) = match.groups()
    origin, destination = context
    if origin is None or destination is None:
        raise ValueError("missing route context")
    first, last = _resolve_dates(sd, sm, ed, em, bounds)
    digits = days.replace(" ", "")
    if len(set(digits)) != len(digits) or digits != "".join(sorted(digits)):
        raise ValueError("invalid weekdays")
    if not hours and not minutes:
        raise ValueError("missing elapsed duration")
    if int(minutes or 0) >= 60:
        raise ValueError("invalid duration minute component")
    return ScheduleRecord(
        origin=origin,
        destination=destination,
        valid_from=first,
        valid_to=last,
        weekdays=sum(1 << (int(digit) - 1) for digit in digits),
        departure_minute=_parse_clock(departure),
        arrival_minute=_parse_clock(arrival),
        arrival_day_offset=int((offset or "0").removeprefix("+")),
        flight_number=flight,
        codeshare=bool(star),
        aircraft=aircraft,
        duration_minutes=int(hours or 0) * 60 + int(minutes or 0),
    )


def iter_pdf_batches(
    path: str | Path, options: ParseOptions = ParseOptions()
) -> Iterator[tuple[ScheduleRecord, ...]]:
    """Yield bounded batches of validated schedules using sequential PDFium calls."""
    path = Path(path)
    if path.stat().st_size > options.max_file_bytes:
        raise ResourceLimitError("PDF exceeds configured byte limit.")
    try:
        document = pdfium.PdfDocument(path)
    except (pdfium.PdfiumError, OSError, ValueError) as exc:
        raise PdfParseError("PDF cannot be opened or requires a password.") from exc
    records: list[ScheduleRecord] = []
    contexts: list[list[str | None]] = [[None, None], [None, None]]
    bounds = (
        (options.valid_from, options.valid_to)
        if options.valid_from is not None and options.valid_to is not None
        else None
    )
    recognized = False
    try:
        if len(document) > options.max_pages:
            raise ResourceLimitError("PDF exceeds configured page limit.")
        for index in range(len(document)):
            page = None
            textpage = None
            try:
                page = document[index]
                textpage = page.get_textpage()
                count = textpage.count_chars()
                if count > options.max_page_chars:
                    raise ResourceLimitError(
                        f"Page {index + 1} exceeds configured character limit."
                    )
                if count == 0:
                    _LOGGER.warning("Skipping page %d: no extractable text.", index + 1)
                    contexts = [[None, None], [None, None]]
                    continue
                width, height = page.get_size()
                if index == 0 and bounds is None:
                    cover = _COVER.search(textpage.get_text_bounded())
                    if cover:
                        day, month, year, end_day, end_month, end_year = cover.groups()
                        try:
                            bounds = (
                                date(int(year), _MONTHS[month], int(day)),
                                date(int(end_year), _MONTHS[end_month], int(end_day)),
                            )
                        except (ValueError, KeyError) as exc:
                            raise PdfParseError(
                                "Invalid cover validity dates."
                            ) from exc
                        if bounds[0] > bounds[1]:
                            raise PdfParseError("Cover validity dates are not ordered.")
                for column in (1, 2):
                    text = textpage.get_text_bounded(
                        (column - 1) * width / 2, 0, column * width / 2, height
                    )
                    context = contexts[column - 1]
                    for row, line in enumerate(text.splitlines(), 1):
                        line = " ".join(line.split())
                        if not line:
                            continue
                        route = _ROUTE.match(line)
                        if route:
                            recognized = True
                            if route[1] == "FROM":
                                context[:] = [route[2], None]
                            else:
                                context[1] = route[2]
                        elif line.startswith(("FROM:", "TO:")):
                            raise PdfParseError(
                                f"Invalid route heading on page {index + 1}."
                            )
                        elif line.startswith("Consult your travel agent"):
                            continue
                        elif re.match(r"^\d{2}\s+\w{3}\s*-", line):
                            match = _ROW.fullmatch(line)
                            if match is None:
                                raise UnsupportedLayoutError(
                                    f"Unsupported schedule row on page {index + 1}, "
                                    f"column {column}, row {row}."
                                )
                            try:
                                record = _build_record(match, context, bounds)
                            except ValueError as exc:
                                raise PdfParseError(
                                    f"Invalid schedule values on page {index + 1}, "
                                    f"column {column}, row {row}."
                                ) from exc
                            records.append(record)
                        elif (
                            all(context)
                            and line not in _HEADERS
                            and not line.startswith("Operated by:")
                        ):
                            raise UnsupportedLayoutError(
                                f"Unsupported table content on page {index + 1}."
                            )
                        if len(records) >= options.batch_size:
                            yield tuple(records)
                            records.clear()
            except pdfium.PdfiumError as exc:
                raise PdfParseError(f"Unable to extract page {index + 1}.") from exc
            finally:
                if textpage is not None:
                    textpage.close()
                if page is not None:
                    page.close()
        if not recognized:
            raise UnsupportedLayoutError("No supported timetable routes found.")
    finally:
        document.close()
    if records:
        yield tuple(records)
