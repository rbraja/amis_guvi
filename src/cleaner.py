
import pandas as pd
import re


def fix_emails(df):

    email_cols = [

        col

        for col in df.columns

        if "email" in col.lower()

    ]

    pattern = (
        r'^[A-Za-z0-9._%+-]+'
        r'@[A-Za-z0-9.-]+'
        r'\.[A-Za-z]{2,}$'
    )

    log = []

    for col in email_cols:

        mask = df[col].astype(str).str.match(
            pattern,
            na=False
        )

        invalid_count = (~mask).sum()

        if invalid_count > 0:

            df.loc[
                ~mask,
                col
            ] = "invalid@email.com"

            log.append({

                "Issue":
                    f"Invalid Email - {col}",

                "Count":
                    invalid_count,

                "Action":
                    "Corrected"

            })

    return df, log

def fix_dates(df):

    date_cols = [

        col

        for col in df.columns

        if "date" in col.lower()

    ]

    log = []

    for col in date_cols:

        before_invalid = (

            pd.to_datetime(

                df[col],

                errors="coerce"

            )

            .isnull()

            .sum()

        )

        df[col] = pd.to_datetime(

            df[col],

            errors="coerce"

        )

        log.append({

            "Issue":
                f"Invalid Date - {col}",

            "Count":
                before_invalid,

            "Action":
                "Converted"

        })

    return df, log
def fix_resolution_time(df):

    log = []

    if "resolution_time_hours" in df.columns:

        invalid_count = (

            df["resolution_time_hours"] < 0

        ).sum()

        df.loc[
            df["resolution_time_hours"] < 0,

            "resolution_time_hours"

        ] = 0

        log.append({

            "Issue":
                "Negative Resolution Time",

            "Count":
                invalid_count,

            "Action":
                "Replaced With 0"

        })

    return df, log

def fix_category_codes(df):

    mapping = {

        "ELC":
            "Electrical",

        "ELEC":
            "Electrical",

        "PLMB":
            "Plumbing",

        "HVAC_OLD":
            "HVAC"

    }

    log = []

    for col in df.columns:

        if df[col].dtype == "object":

            count = 0

            for old_value in mapping:

                count += (
                    df[col]
                    .astype(str)
                    .eq(old_value)
                    .sum()
                )

            if count > 0:

                df[col] = df[col].replace(
                    mapping
                )

                log.append({

                    "Issue":
                        f"Legacy Codes - {col}",

                    "Count":
                        count,

                    "Action":
                        "Mapped"

                })

    return df, log