import pandas as pd


def compare_row_counts(
        raw_tables,
        cleaned_tables
):

    result = []

    for table_name in raw_tables.keys():

        raw_rows = len(
            raw_tables[table_name]
        )

        clean_rows = len(
            cleaned_tables[table_name]
        )

        result.append({

            "Table":
                table_name,

            "Raw Rows":
                raw_rows,

            "Cleaned Rows":
                clean_rows,

            "Difference":
                raw_rows-clean_rows

        })

    return pd.DataFrame(result)


def validate_targets(
        cleaned_tables,
        expected_rows
):

    result = []

    for table_name, expected in expected_rows.items():

        actual = len(
            cleaned_tables.get(
                table_name,
                pd.DataFrame()
            )
        )

        status = (
            "PASS"
            if actual == expected
            else "WARNING"
        )

        result.append({

            "Table":
                table_name,

            "Expected":
                expected,

            "Actual":
                actual,

            "Status":
                status

        })

    return pd.DataFrame(result)