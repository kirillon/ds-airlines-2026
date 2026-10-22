"""Printed airline timetable fields; local clocks have no inferred zone."""

from datetime import date
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ClockMinute = Annotated[int, Field(ge=0, le=1439)]
AirportCode = Annotated[str, Field(pattern=r"^[A-Z]{3}$")]


class ScheduleRecord(BaseModel):
    """One advertised service, not evidence of a passenger's travel or identity.

    Weekday bit zero denotes Sunday. Dates are inclusive service-validity bounds.
    Arrival offset counts local calendar days relative to the departure date;
    duration is the published elapsed time, independent of local clock differences.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    origin: AirportCode
    destination: AirportCode
    valid_from: date
    valid_to: date
    weekdays: int = Field(ge=1, le=127)
    departure_minute: ClockMinute
    arrival_minute: ClockMinute
    arrival_day_offset: int = Field(ge=-2, le=2)
    flight_number: str = Field(pattern=r"^[A-Z0-9]{2}[0-9]{1,4}[A-Z]?$")
    aircraft: str = Field(min_length=1, max_length=12)
    duration_minutes: int = Field(gt=0, le=2880)
    codeshare: bool = False

    @field_validator("aircraft")
    @classmethod
    def validate_aircraft(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("aircraft must not be blank")
        return value

    @model_validator(mode="after")
    def validate_schedule(self) -> Self:
        if self.valid_from > self.valid_to:
            raise ValueError("validity dates must be ordered")
        if self.origin == self.destination:
            raise ValueError("route endpoints must differ")
        return self
