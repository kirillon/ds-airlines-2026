import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_AIRPORT_REFERENCE = (
    Path(__file__).resolve().parents[1] / "resources" / "airports.parquet"
)


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    api_keys: set[str]
    airport_reference: Path = DEFAULT_AIRPORT_REFERENCE
    max_file_bytes: int = 128 * 1024 * 1024
    cors_origins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.api_keys, set) or not self.api_keys:
            raise ValueError("Configure AIRLINES_API_KEYS before starting the API")

    @classmethod
    def from_env(cls) -> "Settings":
        keys = json.loads(os.environ.get("AIRLINES_API_KEYS", "[]"))
        if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
            raise ValueError("AIRLINES_API_KEYS must be a JSON array of keys")
        return cls(
            data_dir=Path(os.environ.get("AIRLINES_DATA_DIR", "backend/data"))
            .expanduser()
            .resolve(),
            api_keys=set(keys),
            airport_reference=Path(
                os.environ.get("AIRLINES_AIRPORTS", str(DEFAULT_AIRPORT_REFERENCE))
            )
            .expanduser()
            .resolve(),
            cors_origins=tuple(
                value.strip()
                for value in os.environ.get("AIRLINES_CORS_ORIGINS", "").split(",")
                if value.strip()
            ),
        )
