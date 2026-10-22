from datetime import date, time
from io import StringIO

import pandas as pd


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.mask(df.isin(["", "N/A"]), pd.NA)
    for column in ("PaxBirthDate", "DepartDate", "ArrivalDate"):
        values = df[column]
        parsed = {
            value: date.fromisoformat(value) for value in values.dropna().unique()
        }
        df[column] = values.map(parsed).astype(object)

    for column in ("DepartTime", "ArrivalTime"):
        values = df[column]
        parsed_times = {
            value: time.fromisoformat(value) for value in values.dropna().unique()
        }
        df[column] = values.map(parsed_times).astype(object)

    return df.mask(df.isna(), pd.NA).convert_dtypes()


def parse_tab(tab: str) -> pd.DataFrame:
    df = get_dataframe(tab)
    df = fix_data_types(df)
    return df


def get_dataframe(tab: str) -> pd.DataFrame:
    rows = tab.splitlines()
    if not rows:
        raise ValueError("Input file is empty")
    foundheaders = rows[0].replace(" ", "")
    headers = [
        "PaxName",
        "PaxBirthDate",
        "DepartDate",
        "DepartTime",
        "ArrivalDate",
        "ArrivalTime",
        "Flight",
        "CodeSh",
        "From",
        "Dest",
        "Code",
        "e-Ticket",
        "TravelDoc",
        "Seat",
        "Meal",
        "TrvCls",
        "Fare",
        "Baggage",
        "PaxAdditionalInfo",
        "AgentInfo",
    ]
    if foundheaders != "".join(headers):
        raise ValueError("Wrong headers found, is this the right file?")
    for row in rows[1:]:
        if len(row) != 336:
            raise ValueError("Incorrect row width, is this the right file?")
    widths = [60, 12, 12, 12, 12, 12, 6, 6, 6, 6, 6, 18, 12, 6, 6, 6, 6, 12, 60, 60]

    df = pd.read_fwf(
        StringIO(tab),
        widths=widths,
        names=headers,
        skiprows=1,
        dtype="string",
        keep_default_na=False,
    )
    return df


if __name__ == "__main__":
    raise RuntimeError("Don't run this file")
