import math
from pathlib import Path
from typing import Any

import pandas as pd


def load_airports(reference: Path) -> dict[str, dict[str, Any] | None]:
    if not reference.is_file():
        return {}
    frame = pd.read_parquet(reference).astype(object)
    rows = frame.where(frame.notna(), None).to_dict(orient="records")
    airports: dict[str, dict[str, Any] | None] = {}
    for fields in (("iata_code",), ("icao_code", "ident")):
        candidates: dict[str, dict[str, Any] | None] = {}
        for row in rows:
            latitude = row.get("latitude_deg", row.get("latitude"))
            longitude = row.get("longitude_deg", row.get("longitude"))
            if not all(
                isinstance(value, (float, int)) and math.isfinite(value)
                for value in (latitude, longitude)
            ):
                continue
            if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                continue
            for field in fields:
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    continue
                code = value.strip().upper()
                info = dict(
                    code=code,
                    latitude=latitude,
                    longitude=longitude,
                    name=row.get("name"),
                )
                if code in candidates and candidates[code] != info:
                    candidates[code] = None
                else:
                    candidates.setdefault(code, info)
        for code, candidate in candidates.items():
            airports.setdefault(code, candidate)
    return airports
