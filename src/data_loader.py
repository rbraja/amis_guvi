import pandas as pd
import os

def load_all_tables(folder_path):

    tables = {}

    if not os.path.exists(folder_path):
        return tables

    for file in os.listdir(folder_path):

        if file.endswith(".csv"):

            table_name = file.replace(".csv", "")

            file_path = os.path.join(
                folder_path,
                file
            )

            tables[table_name] = pd.read_csv(
                file_path
            )

    return tables