import pandas as pd

def profile_tables(tables):

    result = []

    for table_name, df in tables.items():

        result.append({

            "Table Name":
                table_name,

            "Rows":
                df.shape[0],

            "Columns":
                df.shape[1],

            "Missing Values":
                df.isnull().sum().sum(),

            "Duplicate Rows":
                df.duplicated().sum(),

            "Invalid Emails":
                count_invalid_emails(df),

            "Invalid Dates":
                count_invalid_dates(df),

            "Invalid Phones":
                count_invalid_phones(df),

            "Negative Resolution":
                count_negative_resolution(df)

        })

    return pd.DataFrame(result)

def count_invalid_emails(df):

    invalid = 0

    email_cols = [

        c for c in df.columns

        if "email" in c.lower()

    ]

    pattern = (
        r'^[A-Za-z0-9._%+-]+'
        r'@[A-Za-z0-9.-]+'
        r'\.[A-Za-z]{2,}$'
    )

    for col in email_cols:

        invalid += (

            ~df[col]
            .astype(str)
            .str.match(pattern)

        ).sum()

    return invalid
def count_invalid_dates(df):

    invalid = 0

    date_cols = [

        c for c in df.columns

        if "date" in c.lower()
    ]

    for col in date_cols:

        converted = pd.to_datetime(

            df[col],

            errors="coerce"
        )

        invalid += converted.isnull().sum()

    return invalid
def count_invalid_phones(df):

    invalid = 0

    phone_cols = [

        c for c in df.columns

        if "phone" in c.lower()
    ]

    for col in phone_cols:

        invalid += (

            ~df[col]
            .astype(str)
            .str.match(r'^\d{10}$')

        ).sum()

    return invalid

def count_negative_resolution(df):

    count = 0

    if "resolution_time_hours" in df.columns:

        count = (

            df["resolution_time_hours"] < 0

        ).sum()

    return count