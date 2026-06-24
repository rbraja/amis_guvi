import pandas as pd


def create_season(month):

    if month in [12, 1, 2]:
        return "Winter"

    elif month in [3, 4, 5]:
        return "Summer"

    elif month in [6, 7, 8]:
        return "Monsoon"

    else:
        return "Autumn"


def add_season_flag(df):

    df["request_date"] = pd.to_datetime(
        df["request_date"]
    )

    df["season_flag"] = (
        df["request_date"]
        .dt.month
        .apply(create_season)
    )

    return df


def add_complaint_hour(df):

    df["complaint_hour"] = (
        pd.to_datetime(
            df["request_date"]
        ).dt.hour
    )

    return df


def add_repeat_complaint(df):

    complaint_count = (

        df.groupby("resident_id_x")

        ["request_id"]

        .transform("count")

    )

    df["is_repeat_complaint"] = (

        complaint_count > 1

    )

    return df


def add_asset_failure_count(
        master_df,
        asset_failure_df
):

    failure_count = (

        asset_failure_df

        .groupby("asset_id")

        .size()

        .reset_index(
            name="asset_failure_count"
        )

    )

    master_df = master_df.merge(
        failure_count,
        on="asset_id",
        how="left"
    )

    master_df[
        "asset_failure_count"
    ] = master_df[
        "asset_failure_count"
    ].fillna(0)

    return master_df