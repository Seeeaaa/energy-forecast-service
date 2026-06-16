from pandas import DataFrame, Series, Timedelta, Timestamp, DateOffset
from pandas.core.groupby.generic import DataFrameGroupBy
import pandas as pd
import numpy as np


def get_lag(
    df: DataFrame,
    dt: str = "datetime",
    lag: int = 2,
    columns: list[str] = ["target"],
) -> DataFrame:
    """
    Shift 'dt' column by 'lag' days and rename the 'c' column.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    dt : str
        Name of the datetime column to be shifted.
    lag : int
        Number of days to shift (must be 2 or greater).
    columns : list[str]
        List of columns to rename.

    Returns
    -------
    DataFrame
        DataFrame with the shifted datetime column and renamed target
        column.

    Raises
    ------
    ValueError
        If 'lag' is less than 2.
    KeyError
        If any column from 'columns' is not in the DataFrame.
    """
    if lag < 2:
        raise ValueError(f"'lag' must be at least 2 days, got {lag}")

    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")

    return df.assign(**{dt: df[dt] + Timedelta(days=lag)}).rename(
        columns={col: f"{lag}d_lag_{col}" for col in columns}
    )


def get_moving_average(
    sorted_dfgb: DataFrameGroupBy,
    columns: list[str],
    window: int = 24,
    min_periods: int | None = None,
) -> DataFrame:
    """
    Compute rolling mean for specified columns of a grouped DataFrame
    and shift the datetime column by 48 hours.

    Parameters
    ----------
    sorted_dfgb : DataFrameGroupBy
        Grouped DataFrame (result of df.groupby(..., as_index=False)),
        where the original DataFrame was sorted by the datetime64[ns]
        index.
    columns : list[str]
        List of columns to aggregate.
    window : int
        Rolling window size in hours (min_periods=window).
    min_periods : int | None
        Minimum number of observations in the window required to have a
        value otherwise None.

    Returns
    -------
    DataFrame
        DataFrame containing:
        - all grouping columns,
        - the datetime column, shifted by 48 hours,
        - a new columns with the rolling mean.
    """
    txt = "h_ma_2d_lag_"
    missing = [c for c in columns if c not in sorted_dfgb.obj.columns]
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")

    # Store original dtypes
    original_dtypes = {
        f"{window}{txt}{c}": sorted_dfgb.obj[c].dtype for c in columns
    }
    df_rolled = (
        sorted_dfgb[columns]
        .rolling(
            Timedelta(f"{window} h"),
            min_periods=min_periods,
            closed="left",
        )
        .mean()
        .reset_index()
    )
    # df_rolled.iloc[:, 0] += Timedelta(hours=48)  # Shift datetime
    df_rolled["datetime"] += Timedelta(hours=48)
    df_rolled = df_rolled.rename(
        columns={c: f"{window}{txt}{c}" for c in columns}
    )
    df_rolled = df_rolled.astype(original_dtypes)
    return df_rolled


def add_dst_flag(df: DataFrame, datetime_col: str = "datetime") -> DataFrame:
    """
    Add a boolean 'dst' column indicating timestamps that fall within
    the predefined DST intervals for 2021-2023.
    """
    df["dst"] = (
        (df[datetime_col] < "2021-10-31 03:00:00")
        | (
            (df[datetime_col] >= "2022-03-27 03:00:00")
            & (df[datetime_col] < "2022-10-30 03:00:00")
        )
        | (df[datetime_col] >= "2023-03-26 03:00:00")
    )
    return df


def add_cyclic_datetime_features(
    df: DataFrame, datetime_col: str = "datetime", drop_raw: bool = True
) -> DataFrame:
    """
    Extract and encode cyclical datetime features.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame that contain datetime column.
    datetime_col : str
        Name of the datetime column to process.
    drop_raw : bool
        If True, drop the intermediate integer datetime columns (e.g.
        hour, weekday) after encoding.

    Returns
    -------
    DataFrame
        Same DataFrame with sin-cos features.

    Raises
    ------
    KeyError
        If 'datetime_col' is not in DataFrame.
    """
    if datetime_col not in df:
        raise KeyError(f"Column {datetime_col} not in DataFrame")
    df = df.copy()
    dt = pd.to_datetime(df[datetime_col])

    df["hour"] = dt.dt.hour
    df["weekday"] = dt.dt.weekday
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["quarter"] = dt.dt.quarter

    # Cyclic format
    for col, period in [
        ("hour", 24),
        ("weekday", 7),
        ("day_of_month", 30.4),
        ("month", 12),
        ("day_of_year", 365),
        ("week_of_year", 52),
        ("quarter", 4),
    ]:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period).astype(
            "float32"
        )
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period).astype(
            "float32"
        )

    if drop_raw:
        df = df.drop(
            columns=[
                "hour",
                "weekday",
                "day_of_month",
                "month",
                "day_of_year",
                "week_of_year",
                "quarter",
            ]
        )

    return df
