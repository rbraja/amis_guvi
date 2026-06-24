import pandas as pd


def remove_dataset_version(df):

    if "dataset_version" in df.columns:
        df = df.drop(
            columns=["dataset_version"]
        )

    return df


def integrate_tables(tables):

    master_df = remove_dataset_version(
        tables["maintenance_requests"].copy()
    )

    if "apartments" in tables:

        apartments = remove_dataset_version(
            tables["apartments"]
        )

        master_df = master_df.merge(
            apartments,
            on="apartment_id",
            how="left"
        )

    if "residents" in tables:

        residents = remove_dataset_version(
            tables["residents"]
        )

        master_df = master_df.merge(
            residents,
            on="resident_id",
            how="left"
        )

    if "technicians" in tables:

        technicians = remove_dataset_version(
            tables["technicians"]
        )

        master_df = master_df.merge(
            technicians,
            left_on="assigned_technician_id",
            right_on="technician_id",
            how="left"
        )

    if "building_assets" in tables:

        assets = remove_dataset_version(
            tables["building_assets"]
        )

        master_df = master_df.merge(
            assets,
            on="asset_id",
            how="left"
        )

    if "service_logs" in tables:

        service_logs = remove_dataset_version(
            tables["service_logs"]
        )

        master_df = master_df.merge(
            service_logs,
            on="request_id",
            how="left"
        )

    if "resident_feedback" in tables:

        feedback = remove_dataset_version(
            tables["resident_feedback"]
        )

        master_df = master_df.merge(
            feedback,
            on="request_id",
            how="left"
        )

    return master_df