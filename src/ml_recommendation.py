import pandas as pd
def recommend_technicians(
    df,
    category
):

    tech_df = (

        df.groupby(

            [

                "technician_name",

                "category"

            ]

        )

        .agg(

            average_rating=(

                "rating_y",

                "mean"

            ),

            average_resolution_time=(

                "resolution_time_hours",

                "mean"

            ),

            completed_jobs=(

                "request_id",

                "count"

            )

        )

        .reset_index()

    )

    tech_df["score"] = (

        tech_df["average_rating"] * 10

    ) + (

        tech_df["completed_jobs"] * 0.5

    ) - (

        tech_df["average_resolution_time"] * 0.2

    )

    recommendations = (

        tech_df[

            tech_df["category"] == category

        ]

        .sort_values(

            "score",

            ascending=False

        )

        .head(10)

    )

    return recommendations