from typing import Any, Literal

from pydantic import BaseModel


class Page(BaseModel):
    total: int
    offset: int
    limit: int
    items: list[dict[str, Any]]


class RelatedRows(BaseModel):
    total: int
    items: list[dict[str, Any]]
    table: str
    available: bool
    linkage: Literal["lineage", "representative_only"]
