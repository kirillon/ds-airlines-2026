"""Conservative passenger linkage before behavioral feature construction.

Hypothesis: independently corroborated document, ticket and full-card keys improve
history coverage. Names alone remain uncertain proxies; linkage never uses labels.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from airlines_ml.cleaning import (
    clean_text,
    normalize_identifier,
    normalize_name,
)
from airlines_ml.geography import normalize_country
from airlines_ml.inputs import LoadedSource, build_tab_identity
from airlines_ml.recovery import recover_event_fields


@dataclass
class ResolvedData:
    source: LoadedSource
    passengers: pl.DataFrame
    passenger_sources: pl.DataFrame


_NODE_SCHEMA = {
    "id": pl.String,
    "source": pl.String,
    "name": pl.String,
    "name_key": pl.String,
    "alias": pl.String,
    "dob": pl.Date,
    "document": pl.String,
    "document_country": pl.String,
    "ticket": pl.String,
    "card": pl.String,
}


def _read(path: Path, columns: list[str]) -> pl.DataFrame:
    scan = pl.scan_parquet(path)
    available = scan.collect_schema().names()
    return scan.select([column for column in columns if column in available]).collect()


def _column(
    frame: pl.DataFrame, name: str, dtype: pl.DataType | type[pl.DataType] = pl.String
) -> pl.Expr:
    return (
        pl.col(name).cast(dtype) if name in frame.columns else pl.lit(None, dtype=dtype)
    )


def _build_nodes(
    frame: pl.DataFrame,
    source: str,
    identifier: pl.Expr,
    name: pl.Expr,
    *,
    document: pl.Expr | None = None,
    document_country: pl.Expr | None = None,
    ticket: pl.Expr | None = None,
    card: pl.Expr | None = None,
    dob: pl.Expr | None = None,
) -> pl.DataFrame:
    missing = pl.lit(None, dtype=pl.String)
    result = (
        frame.select(
            identifier.alias("id"),
            pl.lit(source).alias("source"),
            pl.when(clean_text(name).str.contains("\\p{L}"))
            .then(clean_text(name))
            .otherwise(None)
            .alias("name"),
            (dob if dob is not None else pl.lit(None, dtype=pl.Date)).alias("dob"),
            (normalize_identifier(document) if document is not None else missing).alias(
                "document"
            ),
            (
                pl.coalesce(
                    normalize_country(document_country), clean_text(document_country)
                )
                if document_country is not None
                else missing
            ).alias("document_country"),
            (normalize_identifier(ticket) if ticket is not None else missing).alias(
                "ticket"
            ),
            (normalize_identifier(card) if card is not None else missing).alias("card"),
        )
        .filter(pl.col("id").is_not_null())
        .with_columns(normalize_name(pl.col("name")).alias("name_key"))
    )
    words = pl.col("name_key").str.split(" ")
    alias = pl.when(words.list.len() >= 2).then(
        pl.concat_str(
            [words.list.get(1, null_on_oob=True), words.list.first()], separator=" "
        )
    )
    return result.with_columns(alias.alias("alias")).select(list(_NODE_SCHEMA)).unique()


def _has_full_name() -> pl.Expr:
    words = clean_text(pl.col("name")).str.replace_all("['-]", "").str.split(" ")
    first = words.list.get(0, null_on_oob=True).str.contains(r"^\p{L}{2,}$")
    second = words.list.get(1, null_on_oob=True).str.contains(r"^\p{L}{2,}$")
    # XLSX uses firstname, optional patronymic initial, then surname.
    xlsx = (
        (pl.col("source") == "xlsx")
        & (words.list.len() == 3)
        & words.list.last().str.contains(r"^\p{L}{2,}$")
    )
    return (first & (second | xlsx)).fill_null(False)


def _extract_source(loaded: LoadedSource) -> tuple[pl.DataFrame, pl.DataFrame]:
    card: pl.Expr | None
    source = loaded.name
    folder = loaded.input_files[0].parent
    if source == "tab":
        raw = _read(
            loaded.input_files[0],
            [
                "TravelDoc",
                "PaxName",
                "PaxBirthDate",
                "e-Ticket",
                "From",
                "issuing_country",
                "document_country",
            ],
        )
        identifier = build_tab_identity(raw)
        nodes = _build_nodes(
            raw,
            source,
            identifier,
            _column(raw, "PaxName"),
            document=pl.col("TravelDoc"),
            document_country=pl.coalesce(
                clean_text(_column(raw, "issuing_country")),
                clean_text(_column(raw, "document_country")),
            ),
            ticket=_column(raw, "e-Ticket"),
            dob=_column(raw, "PaxBirthDate", pl.Date),
        )
        events = (
            loaded.frame.with_columns(_column(raw, "From").alias("Origin"))
            if "From" not in raw.columns
            else loaded.frame.with_columns(raw.get_column("From").alias("Origin"))
        )
    elif source in {"json", "xml"}:
        prefix = "xml_" if source == "xml" else ""
        raw = _read(
            folder / f"{prefix}profiles.parquet",
            ["profile_index", "first_name", "last_name", "middle_name"],
        )
        name = pl.concat_str(
            [
                clean_text(_column(raw, "last_name")),
                clean_text(_column(raw, "first_name")),
                clean_text(_column(raw, "middle_name")),
            ],
            separator=" ",
            ignore_nulls=True,
        )
        nodes = _build_nodes(
            raw,
            source,
            pl.lit(f"{source}:profile:") + pl.col("profile_index").cast(pl.String),
            name,
        )
        cards = _read(
            folder / f"{prefix}cards.parquet",
            ["profile_index", "program", "number", "card_number"],
        )
        card = (
            _column(cards, "card_number")
            if source == "xml"
            else pl.concat_str([_column(cards, "program"), _column(cards, "number")])
        )
        cards = cards.select(
            (
                pl.lit(f"{source}:profile:") + pl.col("profile_index").cast(pl.String)
            ).alias("id"),
            normalize_identifier(card).alias("card"),
        )
        nodes = (
            nodes.drop("card")
            .join(cards, on="id", how="left")
            .select(list(_NODE_SCHEMA))
        )
        raw_events = _read(
            folder / f"{prefix}flights.parquet",
            ["departure_airport", "departure_country", "arrival_country"],
        )
        events = loaded.frame.with_columns(
            raw_events.get_column("departure_airport").alias("Origin")
            if "departure_airport" in raw_events.columns
            else pl.lit(None, dtype=pl.String).alias("Origin"),
            *(
                raw_events.get_column(raw_name).alias(target)
                if raw_name in raw_events.columns
                else pl.lit(None, dtype=pl.String).alias(target)
                for raw_name, target in [
                    ("departure_country", "OriginCountry"),
                    ("arrival_country", "DestinationCountry"),
                ]
            ),
        )
    elif source == "xlsx":
        raw = _read(
            loaded.input_files[0], ["passenger_name", "ticket_number", "origin"]
        ).with_row_index("row")
        nodes = _build_nodes(
            raw,
            source,
            pl.lit("xlsx:record:") + pl.col("row").cast(pl.String),
            _column(raw, "passenger_name"),
            ticket=_column(raw, "ticket_number"),
        )
        events = loaded.frame.with_columns(
            (pl.lit("xlsx:record:") + pl.int_range(pl.len()).cast(pl.String)).alias(
                "PassengerDocument"
            ),
            raw.get_column("origin").alias("Origin")
            if "origin" in raw.columns
            else pl.lit(None, dtype=pl.String).alias("Origin"),
        )
    else:
        raw = loaded.identities
        yaml_card = (
            pl.col("PassengerDocument").str.strip_prefix("yaml:card:")
            if source == "yaml"
            else None
        )
        nodes = _build_nodes(
            raw,
            source,
            pl.col("PassengerDocument"),
            pl.lit(None, dtype=pl.String),
            card=yaml_card,
        )
        events = loaded.frame
        if source == "yaml":
            flights = _read(
                folder / "yaml_flights.parquet",
                ["date", "flight", "arrival_airport", "departure_airport"],
            )
            if "departure_airport" in flights.columns:
                origins = flights.select(
                    pl.col("date").alias("FlightDate"),
                    pl.col("flight").alias("FlightNumber"),
                    pl.col("arrival_airport").alias("Destination"),
                    pl.col("departure_airport").alias("Origin"),
                )
                events = events.join(
                    origins,
                    on=["FlightDate", "FlightNumber", "Destination"],
                    how="left",
                    validate="m:1",
                )
    return nodes, events


def _compute_fi_key(name_key: str | None) -> str | None:
    """Extract surname+firstname key for relaxed matching."""
    if not name_key:
        return None
    words = name_key.split()
    if len(words) < 2 or any(len(word) < 2 or not word.isalpha() for word in words[:2]):
        return None
    return " ".join(words[:2])


def _patronymics_compatible(left_tail: list[str], right_tail: list[str]) -> bool:
    """Check if patronymic parts are compatible (abbreviation vs full, or missing)."""
    if not left_tail or not right_tail:
        return True
    if len(left_tail) != len(right_tail):
        return False
    for left, right in zip(left_tail, right_tail, strict=False):
        if left == right:
            continue
        if min(len(left), len(right)) == 1 and left[0] == right[0]:
            continue
        return False
    return True


def _names_are_compatible(left: str, right: str) -> bool:
    left_words, right_words = left.split(), right.split()
    if left_words[:2] == right_words[:2]:
        return _patronymics_compatible(left_words[2:], right_words[2:])
    return False


def _compute_document_conflicts(nodes: pl.DataFrame) -> pl.DataFrame:
    evidence = (
        nodes.filter(pl.col("document").is_not_null() & _has_full_name())
        .select("passenger_id", "document_country", "document", "name_key")
        .unique()
    )
    conflicts: dict[str, int] = defaultdict(int)
    repeated = (
        evidence.group_by("document_country", "document")
        .agg(pl.col("name_key").n_unique().alias("name_count"))
        .filter(pl.col("name_count") > 1)
    )
    evidence = evidence.join(
        repeated.select("document_country", "document"),
        on=["document_country", "document"],
        how="inner",
        nulls_equal=True,
    )
    for group in evidence.partition_by(
        ["document_country", "document"], maintain_order=False
    ):
        names = group.get_column("name_key").unique().to_list()
        if any(
            not _names_are_compatible(left, right)
            for offset, left in enumerate(names)
            for right in names[offset + 1 :]
        ):
            for passenger_id in group.get_column("passenger_id").unique():
                conflicts[passenger_id] += 1
    return pl.DataFrame(
        {
            "passenger_id": list(conflicts),
            "document_conflict_count": list(conflicts.values()),
        },
        schema={"passenger_id": pl.String, "document_conflict_count": pl.Int64},
    )


def resolve_passengers(sources: list[LoadedSource]) -> ResolvedData:
    """Resolve unique strong keys; keep ambiguous identities separate."""
    if not sources:
        raise ValueError("At least one passenger source is required")
    extracted = [_extract_source(source) for source in sources]
    nodes = pl.concat([pair[0] for pair in extracted], how="vertical_relaxed").unique()
    ids = sorted(nodes.get_column("id").unique().to_list())
    index = {value: position for position, value in enumerate(ids)}
    parent = list(range(len(ids)))
    births: dict[int, set] = defaultdict(set)
    for identifier, dob in nodes.select("id", "dob").unique().iter_rows():
        if dob is not None:
            births[index[identifier]].add(dob)
    anchored: set[int] = set()
    component_names: dict[int, set[str]] = defaultdict(set)
    for identifier, name_key in nodes.select("id", "name_key").unique().iter_rows():
        if name_key is not None:
            component_names[index[identifier]].add(name_key)

    def find(position: int) -> int:
        while parent[position] != position:
            parent[position] = parent[parent[position]]
            position = parent[position]
        return position

    def merge(left: str, right: str, evidence: str) -> None:
        a, b = (find(index[left]), find(index[right]))
        if a == b:
            return
        if len(births[a] | births[b]) > 1:
            return
        a, b = sorted((a, b))
        parent[b] = a
        if a in anchored or b in anchored:
            anchored.discard(b)
            anchored.add(a)
        births[a].update(births.pop(b, set()))
        component_names[a].update(component_names.pop(b, set()))
        if evidence in {"document", "ticket", "card"}:
            anchored.add(a)

    for field in ["document", "ticket", "card"]:
        keys = (
            nodes.filter(pl.col(field).is_not_null())
            .select(
                field,
                "source",
                "id",
                *(["document_country"] if field == "document" else []),
            )
            .unique()
        )
        if field == "document":
            keys = keys.with_columns(
                pl.struct("document_country", "document").alias("document")
            ).drop("document_country")
        repeated = keys.group_by(field).len().filter(pl.col("len") > 1).select(field)
        keys = keys.join(repeated, on=field, how="inner")
        for key, owner_sources, owner_ids in (
            keys.group_by(field).agg(pl.col("source"), pl.col("id")).iter_rows()
        ):
            rows = zip(owner_sources, owner_ids, strict=True)
            source_owners: dict[str, list[str]] = defaultdict(list)
            for source, identifier in rows:
                source_owners[source].append(identifier)
            unique_ids = sorted(set(owner_ids))
            if field == "card" and any(
                len(values) > 1 for values in source_owners.values()
            ):
                continue
            if field == "ticket" and len(source_owners.get("tab", [])) != 1:
                continue
            for identifier in unique_ids[1:]:
                merge(unique_ids[0], identifier, field)
    named = (
        nodes.filter(pl.col("name_key").is_not_null() & _has_full_name())
        .select("id", "name_key", "alias", "dob")
        .unique()
    )
    for members_list in (
        named.filter(pl.col("dob").is_not_null())
        .group_by("name_key", "dob")
        .agg(pl.col("id").unique())
        .get_column("id")
    ):
        members = sorted(members_list.to_list())
        for identifier in members[1:]:
            merge(members[0], identifier, "name_and_birth_date")
    named_no_dob = named.filter(pl.col("dob").is_null())
    fi_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for identifier, name_key in named_no_dob.select("id", "name_key").iter_rows():
        fi_key = _compute_fi_key(name_key)
        if fi_key:
            fi_groups[fi_key].append((identifier, name_key))
    for fi_key, fi_members in fi_groups.items():
        members = sorted({identifier for identifier, _ in fi_members})
        roots = {find(index[identifier]) for identifier in members}
        if len(roots) < 2:
            continue
        targets = roots & {find(root) for root in roots if root in anchored}
        if len(targets) != 1:
            continue
        target = next(iter(targets))
        tails = {
            tuple(name.split()[2:])
            for root in roots
            for name in component_names[root]
            if _compute_fi_key(name) == fi_key
        }
        suffixes = sorted(tails)
        compatible = all(
            (
                _patronymics_compatible(list(left), list(right))
                for offset, left in enumerate(suffixes)
                for right in suffixes[offset + 1 :]
            )
        )
        for identifier in members:
            if find(index[identifier]) == find(target):
                continue
            if compatible:
                merge(ids[find(target)], identifier, "fi_name_anchored")
    anchored = {find(root) for root in anchored}
    anchor_roots = anchored
    for aliases, member_ids in (
        named.filter(pl.col("alias").is_not_null())
        .group_by("alias")
        .agg(pl.col("name_key"), pl.col("id"))
        .select("name_key", "id")
        .iter_rows()
    ):
        members = sorted(set(member_ids))
        root_names: dict[int, set[str]] = defaultdict(set)
        for identifier, name in zip(member_ids, aliases, strict=True):
            root_names[find(index[identifier])].add(name)
        targets = set(root_names) & anchored
        for identifier in members:
            root = find(index[identifier])
            if root in targets:
                continue
            compatible_targets = []
            for target in targets:
                compatible = all(
                    _patronymics_compatible(left.split()[2:], right.split()[2:])
                    for left in root_names[root]
                    for right in root_names[target]
                )
                if compatible:
                    compatible_targets.append(target)
            if len(compatible_targets) == 1:
                target = compatible_targets[0]
                merge(ids[target], identifier, "unique_anchored_name_proxy")
                merged_root = find(index[identifier])
                if merged_root != find(target):
                    continue
                combined_names = root_names[root] | root_names[target]
                root_names[merged_root] = combined_names
                targets.discard(target)
                targets.add(merged_root)
    final_anchor_roots = {find(root) for root in anchor_roots}
    xlsx_named = nodes.filter(
        (pl.col("source") == "xlsx")
        & pl.col("name_key").is_not_null()
        & _has_full_name()
    )
    for members_list in (
        xlsx_named.group_by("name_key").agg(pl.col("id").unique()).get_column("id")
    ):
        members = sorted(members_list.to_list())
        if all(find(index[value]) not in final_anchor_roots for value in members):
            for identifier in members[1:]:
                merge(members[0], identifier, "exact_full_name_proxy")
    priority = {"tab": 0, "json": 1, "xml": 2, "yaml": 3, "xlsx": 4}
    components: dict[int, list[str]] = defaultdict(list)
    for identifier in ids:
        components[find(index[identifier])].append(identifier)
    labels = {
        root: "person:"
        + min(members, key=lambda value: (priority.get(value.split(":")[0], 5), value))
        for root, members in components.items()
    }
    mapping = pl.DataFrame(
        {
            "source_passenger_id": ids,
            "passenger_id": [labels[find(index[value])] for value in ids],
            "source": [value.split(":")[0] for value in ids],
        }
    )
    annotated = nodes.join(
        mapping.select(pl.col("source_passenger_id").alias("id"), "passenger_id"),
        on="id",
        how="left",
    )
    name_order = annotated.with_columns(
        pl.col("name").str.split(" ").list.len().fill_null(0).alias("words"),
        pl.col("source").replace_strict(priority, default=5).alias("priority"),
    ).sort(
        ["passenger_id", "words", "priority", "name"],
        descending=[False, True, False, False],
    )
    passengers = name_order.group_by("passenger_id", maintain_order=True).agg(
        pl.col("name").drop_nulls().first().alias("ФИО")
    )
    passengers = passengers.join(
        _compute_document_conflicts(annotated), on="passenger_id", how="left"
    ).with_columns(pl.col("document_conflict_count").fill_null(0))
    events = []
    for _, frame in extracted:
        frame = (
            frame.join(
                mapping.select(
                    pl.col("source_passenger_id").alias("PassengerDocument"),
                    pl.col("passenger_id").alias("resolved"),
                ),
                on="PassengerDocument",
                how="left",
            )
            .drop("PassengerDocument")
            .rename({"resolved": "PassengerDocument"})
        )
        if "Origin" not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias("Origin"))
        frame = frame.select(
            "PassengerDocument",
            "FlightDate",
            "FlightNumber",
            "Origin",
            "Destination",
            _column(frame, "OriginCountry").alias("OriginCountry"),
            _column(frame, "DestinationCountry").alias("DestinationCountry"),
        ).with_columns(
            clean_text(pl.col("FlightNumber"))
            .str.replace_all("\\s+", "")
            .alias("FlightNumber"),
            *(
                clean_text(pl.col(column)).alias(column)
                for column in [
                    "Destination",
                    "Origin",
                    "OriginCountry",
                    "DestinationCountry",
                ]
            ),
        )
        events.append(frame)
    frame = recover_event_fields(pl.concat(events, how="diagonal_relaxed"))
    files = list(
        dict.fromkeys(path for source in sources for path in source.input_files)
    )
    source = LoadedSource(
        "unified",
        frame,
        passengers.select(pl.col("passenger_id").alias("PassengerDocument")),
        files,
    )
    return ResolvedData(source, passengers, mapping)
