from contextlib import closing
from datetime import date, time
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import openpyxl
import pandas as pd


def none_if_na_or_empty(value: object) -> object:
    if isinstance(value, str) and value.strip() in ("", "N/A"):
        return None
    return value


def parse_zip_xlsx(file: Path) -> pd.DataFrame:
    records = []
    with ZipFile(file) as archive:
        for entry in archive.infolist():
            if entry.is_dir() or Path(entry.filename).suffix.lower() != ".xlsx":
                continue
            source_file_date = date.fromisoformat(Path(entry.filename).stem[-10:])
            xlsx = archive.read(entry)
            with closing(
                openpyxl.load_workbook(BytesIO(xlsx), read_only=True)
            ) as workbook:
                for sheet in workbook.worksheets:
                    rows = list(sheet.iter_rows(values_only=True))
                    record = {
                        "sequence": none_if_na_or_empty(rows[0][7]),
                        "passenger_sex": "M"
                        if rows[2][0] == "MR"
                        else "F"
                        if rows[2][0] == "MRS"
                        else None,
                        "passenger_name": rows[2][1],
                        "additional_info": none_if_na_or_empty(rows[2][5]),
                        "class_code": none_if_na_or_empty(rows[2][7]),
                        "flight": rows[4][0],
                        "origin_name": none_if_na_or_empty(rows[4][3]),
                        "destination_name": none_if_na_or_empty(rows[4][7]),
                        "gate": none_if_na_or_empty(rows[6][1]),
                        "origin": rows[6][3],
                        "destination": rows[6][7],
                        "departure_date": date.fromisoformat(rows[8][0]),
                        "departure_time": time.fromisoformat(rows[8][2]),
                        "operator_note": none_if_na_or_empty(rows[8][4]),
                        "seat": none_if_na_or_empty(rows[10][7]),
                        "pnr": rows[12][1],
                        "ticket_number": rows[12][4],
                        "source_file_date": source_file_date,
                    }
                    records.append(record)

    if not records:
        raise ValueError("No xlsx files found, is this the right file?")
    df = pd.DataFrame.from_records(records)
    df["sequence"] = df["sequence"].astype("Int64")
    return df


if __name__ == "__main__":
    raise SystemExit("Don't run this file")
