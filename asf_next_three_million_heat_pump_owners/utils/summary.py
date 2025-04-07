import pandas as pd


def create_single_question_summary_frame(
    data: pd.DataFrame, question_code: str, weight_column: str = "weight"
) -> pd.DataFrame:
    """
    Creates a summary DataFrame for a single question with counts, percentages,
    and weighted counts/percentages based on a given weight column.
    """

    # Counts
    df_counts = (
        data[question_code]
        .value_counts(dropna=False)
        .reindex(data[question_code].cat.categories.append(pd.Index([pd.NA])))
        .to_frame(name="count")
    )

    # Weighted counts
    df_counts["weighted_count"] = (
        data.groupby(question_code, observed=True)[weight_column]
        .sum(min_count=1)
        .reindex(df_counts.index, fill_value=0)
    )

    # Percentages
    df_counts["percentage"] = (df_counts["count"] / df_counts["count"].sum()) * 100

    # Weighted percentages
    total_weighted_count = df_counts["weighted_count"].sum()
    df_counts["weighted_percentage"] = (
        df_counts["weighted_count"] / total_weighted_count
    ) * 100

    # Add total row
    df_counts.loc["Total"] = [
        df_counts["count"].sum(),
        df_counts["weighted_count"].sum(),
        df_counts["percentage"].sum(),
        df_counts["weighted_percentage"].sum(),
    ]

    # Round
    df_counts["percentage"] = df_counts["percentage"].round(2)
    df_counts["weighted_percentage"] = df_counts["weighted_percentage"].round(2)

    return df_counts
