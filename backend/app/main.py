import hmac
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from backend.app.schemas import Page, RelatedRows
from backend.app.settings import Settings
from backend.app.store import ParquetStore

Offset = Annotated[int, Query(ge=0)]
Limit = Annotated[int, Query(ge=1, le=1000)]
PassengerID = Annotated[str, Query(min_length=1, max_length=512)]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = ParquetStore(
        settings.data_dir, settings.airport_reference, settings.max_file_bytes
    )
    api_key = APIKeyHeader(name="X-API-Key", auto_error=False)

    def authenticate(key: Annotated[str | None, Depends(api_key)]) -> None:
        for expected in settings.api_keys:
            if key is not None and hmac.compare_digest(key.encode(), expected.encode()):
                return
        raise HTTPException(401, "Invalid API key")

    app = FastAPI(
        title="Flight analysis API",
        description=".",
        dependencies=[Depends(authenticate)],
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET"],
            allow_headers=["X-API-Key"],
        )

    @app.exception_handler(ValueError)
    @app.exception_handler(KeyError)
    @app.exception_handler(OSError)
    async def handle_data_error(request: Request, exc: Exception):
        if isinstance(exc, KeyError):
            status, detail = 404, "Record or table not found"
        elif isinstance(exc, OSError):
            status, detail = 409, "Data is unavailable"
        else:
            status, detail = 422, "Invalid or incomplete Parquet data"
        return JSONResponse(status_code=status, content={"detail": detail})

    @app.get("/tables", response_model=list[str])
    def list_tables():
        return store.list_tables()

    @app.get("/spies", response_model=Page)
    def list_spies(offset: Offset = 0, limit: Limit = 100):
        return store.list_spies(offset, limit)

    @app.get("/spies/rows", response_model=RelatedRows)
    def list_related(
        passenger_id: PassengerID,
        table: Annotated[str, Query(min_length=1, max_length=64)],
    ):
        return store.list_related(passenger_id, table)

    @app.get("/spies/detail", response_model=dict)
    def get_spy(passenger_id: PassengerID):
        return store.get_spy(passenger_id)

    return app
