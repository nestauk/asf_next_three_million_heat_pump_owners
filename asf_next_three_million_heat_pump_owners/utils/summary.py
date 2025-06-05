import pandas as pd
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot
import matplotlib.pyplot as plt


# --- Plotting ---


def create_stacked_horizontal_bar_chart(
    df: pd.DataFrame, xlabel: str, ylabel: str, fontsize: int, figsize: tuple
) -> matplotlib.pyplot:
    """
    Creates a stacked horizontal bar chart from a DataFrame, with percentage labels displayed on each segment.
    """
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    df.plot(kind="barh", stacked=True, colormap="Dark2", ax=ax)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.legend(loc="center", bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=fontsize)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()

    # Add labels to each segment
    for c in ax.containers:
        for rect in c:
            width = rect.get_width()
            x_position = rect.get_x() + width / 2
            y_position = rect.get_y() + rect.get_height() / 2
            ax.text(
                x_position,
                y_position,
                f"{width:.1f}%",
                ha="center",
                va="center",
                fontsize=fontsize,
                color="white",
            )
    return fig


# --- Summarising single questions ---


def create_single_question_summary_frame(
    data: pd.DataFrame, question_code: str, weight_column: str = "weight"
) -> pd.DataFrame:
    """
    Creates a summary DataFrame for a single question with counts, percentages,
    and weighted counts/percentages based on a given weight column.
    """

    # Counts
    df_summary = (
        data[question_code]
        .value_counts(dropna=False)
        .reindex(data[question_code].cat.categories.append(pd.Index([pd.NA])))
        .to_frame(name="count")
    )

    # Weighted counts
    df_summary["weighted_count"] = (
        data.groupby(question_code, observed=True)[weight_column]
        .sum(min_count=1)
        .reindex(df_summary.index, fill_value=0)
    )

    # Percentages
    df_summary["percentage"] = (df_summary["count"] / df_summary["count"].sum()) * 100

    # Weighted percentages
    total_weighted_count = df_summary["weighted_count"].sum()
    df_summary["weighted_percentage"] = (
        df_summary["weighted_count"] / total_weighted_count
    ) * 100

    # Add total row
    df_summary.loc["Total"] = [
        df_summary["count"].sum(),
        df_summary["weighted_count"].sum(),
        df_summary["percentage"].sum(),
        df_summary["weighted_percentage"].sum(),
    ]

    # Round
    df_summary["percentage"] = df_summary["percentage"].round(2)
    df_summary["weighted_percentage"] = df_summary["weighted_percentage"].round(2)

    return df_summary


def create_nations_summary_tables(
    base_data: pd.DataFrame,
    wales_base_data: pd.DataFrame,
    question_code: str,
    remove_not_asked: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates two summary DataFrames for a single question, one with weighted counts and one with
    weighted percentages. Rows correspond to UK, and nation level data (England, Scotland, Wales and
    Northern Ireland).
    """

    # Individual nation tables
    england = (
        pd.crosstab(
            index=base_data["gornewUK_6"],
            columns=base_data[question_code],
            values=base_data["weight"],
            aggfunc="sum",
        )
        .drop(index=False)
        .rename(index={True: "England"})
    )

    scotland = (
        pd.crosstab(
            index=base_data["gornewUK_8"],
            columns=base_data[question_code],
            values=base_data["weight"],
            aggfunc="sum",
        )
        .drop(index=False)
        .rename(index={True: "Scotland"})
    )

    wales = pd.crosstab(
        index=wales_base_data["gornewUK"],
        columns=wales_base_data[question_code],
        values=wales_base_data["weight"],
        aggfunc="sum",
    )

    northern_ireland = (
        pd.crosstab(
            index=base_data["gornewUK_9"],
            columns=base_data[question_code],
            values=base_data["weight"],
            aggfunc="sum",
        )
        .drop(index=False)
        .rename(index={True: "Northern Ireland"})
    )

    uk = pd.crosstab(
        index=base_data["UK18_Sept_2020_f_0"],
        columns=base_data[question_code],
        values=base_data["weight"],
        aggfunc="sum",
    ).rename(index={True: "UK"})

    # Counts
    df = pd.concat([england, scotland, wales, northern_ireland, uk])
    df = df.reset_index(names="Nation")
    df.set_index("Nation", inplace=True)

    # Proportions
    pct_df = (df.div(df.sum(axis=1), axis=0) * 100).round(1)
    pct_df = pct_df.reset_index(names="Nation")
    pct_df.set_index("Nation", inplace=True)

    if remove_not_asked:
        # Drop redundant columns
        df = df.drop(columns="Not asked")
        pct_df = pct_df.drop(columns="Not asked")

    # Add national total columns
    df["Nation total"] = df.sum(axis=1)

    return df, pct_df


# --- Generating confidence intervals ---


def weighted_props_ci(
    df: pd.DataFrame,
    question_col: str,
    weight_col: str = "weight",
    z_percentile: float = 0.975,
):
    """
    Creates a DataFrame of response proportions for a single question,
    with lower and upper confidence intervals and standard error.
    """

    # z-score for normal distribution
    z = norm.ppf(z_percentile)

    # Weighted counts and proportions
    weighted_counts = df.groupby(question_col, observed=False)[weight_col].sum()
    total_weight = weighted_counts.sum()
    props = weighted_counts / total_weight

    # Effective sample size
    n_eff = (df[weight_col].sum()) ** 2 / (df[weight_col] ** 2).sum()

    # Standard errors
    se = np.sqrt((props * (1 - props)) / n_eff)

    # Confidence intervals
    lower = (props - z * se).clip(lower=0.0)
    upper = (props + z * se).clip(upper=1.0)

    return pd.DataFrame(
        {"Proportion": props, "Lower CI": lower, "Upper CI": upper, "SE": se}
    )


def weighted_props_ci_combine_categories(
    results_df: pd.DataFrame,
    cat1: str,
    cat2: str,
    new_cat_name: str = "Combined",
    z_percentile: float = 0.975,
) -> pd.Series:
    """
    Combines weighted percentages and standard errors of two categories, and computes the resulting
    combined confidence interval and standard error.

    Parameters
    ----------
    results_df : pd.DataFrame
        A DataFrame indexed by category names, containing at least the columns 'Weighted percentage' and 'SE'.
    cat1 : str
        Name of the first category to combine.
    cat2 : str
        Name of the second category to combine.
    new_cat_name : str, optional
        Name to assign to the resulting combined category in the returned Series, by default "Combined".
    z_percentile : float, optional
        Percentile value used to calculate the z-score for the confidence interval, by default 0.975.

    Returns
    -------
    pd.Series
        A Series containing:
            - Weighted percentage
            - Lower CI
            - Upper CI
            - SE
        Named according to `new_cat_name`.
    """

    # z-score for normal distribution
    z = norm.ppf(z_percentile)

    # Proportions and SEs for both categories
    p1, se1 = results_df.loc[cat1, ["Weighted percentage", "SE"]]
    p2, se2 = results_df.loc[cat2, ["Weighted percentage", "SE"]]

    # Combine proportions
    combined_p = p1 + p2

    # Combine standard errors
    combined_se = np.sqrt(se1**2 + se2**2)

    # Combined confidence intervals
    lower = combined_p - z * combined_se
    upper = combined_p + z * combined_se

    combined_result = pd.Series(
        {
            "Weighted percentage": combined_p,
            "Lower CI": lower,
            "Upper CI": upper,
            "SE": combined_se,
        },
        name=new_cat_name,
    )

    return combined_result


# --- Modifying dataframes with confidence intervals ---


def modify_proportion_crosstab_with_ci(
    base_data: pd.DataFrame,
    proportions_df: pd.DataFrame,
    aggregated_responses_column_dict: dict,
    aggregated_responses_index_dict: dict,
    index_survey_code: str,
    z_percentile=0.975,
) -> pd.DataFrame:
    """
    Adds confidence intervals to a crosstab of weighted proportions.

    This function calculates confidence intervals for each proportion in a crosstab, taking into account
    weights and effective sample size. It handles both individual and aggregated response categories
    (in rows and columns). The output is a DataFrame where each proportion is formatted as:
    "value (lower CI - upper CI)".

    Parameters
    ----------
    base_data : pd.DataFrame
        The raw survey dataset containing a column for respondent weights and survey codes.
    proportions_df : pd.DataFrame
        A crosstab DataFrame of weighted percentages (not proportions) indexed by category names.
    aggregated_responses_column_dict : dict
        A dictionary defining aggregated column (response) groupings, where keys are group names
        and values are lists of original column labels to aggregate.
    aggregated_responses_index_dict : dict
        A dictionary defining aggregated index (group) groupings, where keys are group names
        and values are lists of original index labels to aggregate.
    index_survey_code : str
        The column name in `base_data` used to match rows to the `proportions_df` index.
    z_percentile : float, optional
        The percentile value used to calculate the z-score for confidence intervals,
        by default 0.975 (corresponding to ~95% confidence level).

    Returns
    -------
    pd.DataFrame
        A modified version of `proportions_df`, where each cell contains the original
        percentage with its confidence interval in the format: "value (lower - upper)".
    """

    # Dataframes for storing CIs
    ci_lower = pd.DataFrame(
        index=proportions_df.index,
        columns=proportions_df.columns,
    )
    ci_upper = pd.DataFrame(
        index=proportions_df.index,
        columns=proportions_df.columns,
    )

    # Non-aggregated responses
    for index in proportions_df.index:
        if "(net)" not in index and "Total" not in index:
            for column in proportions_df.columns:
                if "(net)" not in column and "Total" not in column:

                    # Weighted counts and proportions
                    group_data = base_data[base_data[index_survey_code] == index]
                    weights = group_data["weight"]
                    total_weight = weights.sum()
                    weighted_proportion = proportions_df.loc[index, column] / 100

                    # Effective sample size
                    n_eff = (total_weight**2) / (weights**2).sum()

                    # Standard error
                    se = np.sqrt(
                        weighted_proportion * (1 - weighted_proportion) / n_eff
                    )

                    # Confidence interval
                    z = norm.ppf(z_percentile)
                    lower = max(0.0, weighted_proportion - z * se)
                    upper = min(1.0, weighted_proportion + z * se)

                    ci_lower.loc[index, column] = round(lower * 100, 1)
                    ci_upper.loc[index, column] = round(upper * 100, 1)

    # Aggregated response columns
    for index in proportions_df.index:
        if "(net)" not in index and "Total" not in index:
            for column in aggregated_responses_column_dict.keys():
                # Weighted counts and proportions
                group_data = base_data[base_data[index_survey_code] == index]
                weights = group_data["weight"]
                total_weight = weights.sum()
                weighted_proportion = proportions_df.loc[index, column] / 100

                # Effective sample size
                n_eff = (total_weight**2) / (weights**2).sum()

                # Standard error
                se = np.sqrt(weighted_proportion * (1 - weighted_proportion) / n_eff)

                # Confidence interval
                z = norm.ppf(z_percentile)
                lower = max(0.0, weighted_proportion - z * se)
                upper = min(1.0, weighted_proportion + z * se)

                ci_lower.loc[index, column] = round(lower * 100, 1)
                ci_upper.loc[index, column] = round(upper * 100, 1)

    # Aggregated response indices
    for index in aggregated_responses_index_dict.keys():
        for column in proportions_df.columns:
            if "Total" not in column:

                # Weighted counts and proportions
                group_data = base_data[
                    base_data[index_survey_code].isin(
                        aggregated_responses_index_dict[index]
                    )
                ]
                weights = group_data["weight"]
                total_weight = weights.sum()
                weighted_proportion = proportions_df.loc[index, column] / 100

                # Effective sample size
                n_eff = (total_weight**2) / (weights**2).sum()

                # Standard error
                se = np.sqrt(weighted_proportion * (1 - weighted_proportion) / n_eff)

                # Confidence interval
                z = norm.ppf(z_percentile)
                lower = max(0.0, weighted_proportion - z * se)
                upper = min(1.0, weighted_proportion + z * se)

                ci_lower.loc[index, column] = round(lower * 100, 1)
                ci_upper.loc[index, column] = round(upper * 100, 1)

    # Modify original comparison table with lower and upper CI values
    proportions_df_with_ci = proportions_df.copy(deep=True)
    proportions_df_with_ci = proportions_df_with_ci.astype(str)
    for index in proportions_df_with_ci.index:
        if index != "Total":
            for column in proportions_df_with_ci.columns:
                if column != "Total":
                    lower = ci_lower.loc[index, column]
                    upper = ci_upper.loc[index, column]
                    proportions_df_with_ci.loc[index, column] = (
                        f"""{proportions_df.loc[index, column].round(1)} ({lower} - {upper})"""
                    )

    return proportions_df_with_ci


# --- Summarising by tenure ---


def generate_tenure_breakdown(
    base_data: pd.DataFrame,
    question: str,
    response_aggregation_dict: dict = None,
    proportions_axis: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates weighted counts and proportions of responses to a question,
    broken down by housing tenure. Optionally aggregates responses and tenure categories.

    Parameters
    ----------
    base_data : pd.DataFrame
        DataFrame containing the survey data, including weights, tenure, and question responses.
    question : str
        The column name of the question to analyse.
    response_aggregation_dict : dict, optional
        Dictionary mapping new column names to lists of response options to aggregate.
        Used to create net categories from multiple response columns.
    proportions_axis : int, default 1
        If 1, calculates row-wise proportions (response breakdown of tenure).
        If 0, calculates column-wise proportions (tenure breakdown of response).

    Returns
    -------
    counts : pd.DataFrame
        Weighted count of responses by tenure and response option.
    proportions : pd.DataFrame
        Proportions table based on counts table, expressed as percentages.
    """

    counts = pd.crosstab(
        index=base_data["profile_house_tenure"],
        columns=base_data[question],
        values=base_data["weight"],
        aggfunc="sum",
    )

    try:
        counts = counts.drop(columns="Not asked")
    except:
        pass

    # Proportions of each row (response breakdown of tenure)
    if proportions_axis == 1:  # Total for each row
        counts.loc["Total"] = counts.sum(axis=0)

        # Aggregate tenure categories
        for tenure in ["Own", "Rent", "Neither"]:
            matching_rows = counts.loc[counts.index.str.contains(tenure)]
            summed_row = matching_rows.sum()
            counts.loc[f"{tenure} (net)"] = summed_row

        proportions = (
            counts.div(counts.sum(axis=proportions_axis), axis=0) * 100
        ).round(1)

        proportions["Total"] = proportions.sum(axis=1)
        counts["Total"] = counts.sum(axis=1)

        if response_aggregation_dict:
            for df in [counts, proportions]:
                for combined_col, source_cols in response_aggregation_dict.items():
                    df[combined_col] = df[source_cols].sum(axis=1)

    # Proportions of each column (tenure breakdown of response)
    elif proportions_axis == 0:  # Total for each column
        counts["Total"] = counts.sum(axis=1)

        if response_aggregation_dict:
            for combined_col, source_cols in response_aggregation_dict.items():
                counts[combined_col] = counts[source_cols].sum(axis=1)

        proportions = (
            counts.div(counts.sum(axis=proportions_axis), axis=1) * 100
        ).round(1)

        counts.loc["Total"] = counts.sum(axis=0)
        proportions.loc["Total"] = proportions.sum(axis=0)

        # Aggregate tenure categories
        for df in [counts, proportions]:
            for tenure in ["Own", "Rent", "Neither"]:
                matching_rows = df.loc[df.index.str.contains(tenure)]
                summed_row = matching_rows.sum()
                df.loc[f"{tenure} (net)"] = summed_row

    else:
        raise ValueError("Invalid axis argument, must be 0 or 1.")

    return counts, proportions


# --- Summarising by income ---


def generate_income_breakdown(
    base_data: pd.DataFrame,
    question: str,
    response_aggregation_dict: dict = None,
    proportions_axis: int = 1,
    aggregate_income: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates weighted counts and proportions of responses to a question,
    broken down by income bracket. Optionally aggregates responses and tenure categories.

    Parameters
    ----------
    base_data : pd.DataFrame
        DataFrame containing the survey data, including weights, income brackets, and question responses.
    question : str
        The column name of the question to analyse.
    response_aggregation_dict : dict, optional
        Dictionary mapping new column names to lists of response options to aggregate.
        Used to create net categories from multiple response columns.
    proportions_axis : int, default 1
        If 1, calculates row-wise proportions (response breakdown of income).
        If 0, calculates column-wise proportions (income breakdown of response).

    Returns
    -------
    counts : pd.DataFrame
        Weighted count of responses by income and response option.
    proportions : pd.DataFrame
        Proportions table based on counts table, expressed as percentages.
    """

    counts = pd.crosstab(
        index=base_data["profile_gross_household"],
        columns=base_data[question],
        values=base_data["weight"],
        aggfunc="sum",
    )

    try:
        counts = counts.drop(columns="Not asked")
    except:
        pass

    # Proportions of each row (response breakdown of tenure)
    if proportions_axis == 1:  # Total for each row
        counts.loc["Total"] = counts.sum(axis=0)

        if aggregate_income:
            # <£10k
            matching_rows = counts.loc[
                ["under £5,000 per year", "£5,000 to £9,999 per year"]
            ]
            summed_row = matching_rows.sum()
            counts.loc["under £10,000 per year (net)"] = summed_row

            # £10,000 to 29,999
            matching_rows = counts.loc[
                [
                    "£10,000 to £14,999 per year",
                    "£15,000 to £19,999 per year",
                    "£20,000 to £24,999 per year",
                    "£25,000 to £29,999 per year",
                ]
            ]
            summed_row = matching_rows.sum()
            counts.loc["£10,000 to £29,999 per year (net)"] = summed_row

            # £30,000 to 49,999
            matching_rows = counts.loc[
                [
                    "£30,000 to £34,999 per year",
                    "£35,000 to £39,999 per year",
                    "£40,000 to £44,999 per year",
                    "£45,000 to £49,999 per year",
                ]
            ]
            summed_row = matching_rows.sum()
            counts.loc["£30,000 to £49,999 per year (net)"] = summed_row

            # £50,000 to 69,999
            matching_rows = counts.loc[
                [
                    "£50,000 to £59,999 per year",
                    "£60,000 to £69,999 per year",
                ]
            ]
            summed_row = matching_rows.sum()
            counts.loc["£50,000 to £69,999 per year (net)"] = summed_row

        proportions = (
            counts.div(counts.sum(axis=proportions_axis), axis=0) * 100
        ).round(1)

        proportions["Total"] = proportions.sum(axis=1)
        counts["Total"] = counts.sum(axis=1)

        # Aggregate response categories
        if response_aggregation_dict:
            for df in [counts, proportions]:
                for combined_col, source_cols in response_aggregation_dict.items():
                    df[combined_col] = df[source_cols].sum(axis=1)

    # Proportions of each column (tenure breakdown of response)
    elif proportions_axis == 0:  # Total for each column
        counts["Total"] = counts.sum(axis=1)

        # Aggregate response categories
        if response_aggregation_dict:
            for combined_col, source_cols in response_aggregation_dict.items():
                counts[combined_col] = counts[source_cols].sum(axis=1)

        proportions = (
            counts.div(counts.sum(axis=proportions_axis), axis=1) * 100
        ).round(1)

        counts.loc["Total"] = counts.sum(axis=0)
        proportions.loc["Total"] = proportions.sum(axis=0)

        if aggregate_income:
            # Aggregate income categories
            for df in [counts, proportions]:
                # <£10k
                matching_rows = df.loc[
                    ["under £5,000 per year", "£5,000 to £9,999 per year"]
                ]
                summed_row = matching_rows.sum()
                df.loc["under £10,000 per year (net)"] = summed_row
                # £10,000 to 29,999
                matching_rows = df.loc[
                    [
                        "£10,000 to £14,999 per year",
                        "£15,000 to £19,999 per year",
                        "£20,000 to £24,999 per year",
                        "£25,000 to £29,999 per year",
                    ]
                ]
                summed_row = matching_rows.sum()
                df.loc["£10,000 to £29,999 per year (net)"] = summed_row
                # £30,000 to 49,999
                matching_rows = df.loc[
                    [
                        "£30,000 to £34,999 per year",
                        "£35,000 to £39,999 per year",
                        "£40,000 to £44,999 per year",
                        "£45,000 to £49,999 per year",
                    ]
                ]
                summed_row = matching_rows.sum()
                df.loc["£30,000 to £49,999 per year (net)"] = summed_row
                # £50,000 to 69,999
                matching_rows = df.loc[
                    [
                        "£50,000 to £59,999 per year",
                        "£60,000 to £69,999 per year",
                    ]
                ]
                summed_row = matching_rows.sum()
                df.loc["50,000 to £69,999 per year (net)"] = summed_row

    else:
        raise ValueError("Invalid axis argument, must be 0 or 1.")

    return counts, proportions


# --- Generating all summary tables for questions with responses to aggregate ---


def generate_q5_outputs(
    base_data, index_survey_code, drop_not_asked=True
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Generates weighted counts, proportions, and confidence intervals for responses to PEN_Q5,
    broken down by a specified survey grouping variable.

    Parameters
    ----------
    base_data : pd.DataFrame
        The survey dataset containing response data, weights, and grouping codes.
    index_survey_code : str
        The name of the column in 'base_data' to group the PEN_Q5 responses by (e.g., age group, region).
    drop_not_asked : bool, optional, default=True
        Whether to drop rows and columns where the response is "Not asked".

    Returns
    -------
    tuple[dict[str, pd.DataFrame], pd.DataFrame]
        - A dictionary containing:
            - "counts": Weighted counts of PEN_Q5 responses by group.
            - "proportions": Proportions of PEN_Q5 responses by group, as percentages.
              Both include net aggregated columns.
        - A DataFrame of proportions with appended 95% confidence intervals for each response by group, with the "Total" column dropped.
    """

    ## Create Q5 response breakdown by response to other chosen question
    q5_tables = {}

    # Counts
    q5_tables["counts"] = pd.crosstab(
        index=base_data[index_survey_code],
        columns=base_data["PEN_Q5"],
        values=base_data["weight"],
        aggfunc="sum",
        margins=True,
        margins_name="Total",
    )

    # Proportions
    q5_tables["proportions"] = pd.crosstab(
        index=base_data[index_survey_code],
        columns=base_data["PEN_Q5"],
        values=base_data["weight"],
        aggfunc="sum",
        normalize="index",
    )
    q5_tables["proportions"]["Total"] = q5_tables["proportions"].sum(axis=1)
    q5_tables["proportions"] = (q5_tables["proportions"] * 100).round(1)

    # Add aggregated responses
    for df in q5_tables.values():
        df["Would consider (net)"] = (
            df["Definitely would consider"] + df["Probably would consider"]
        )
        df["Wouldn't consider (net)"] = (
            df["Definitely wouldn't consider"] + df["Probably wouldn't consider"]
        )
        df.drop(columns="Not asked", inplace=True)
    if drop_not_asked:
        q5_tables["proportions"] = q5_tables["proportions"].drop(index="Not asked")

    ## 95% confidence intervals for proportions
    # Aggregated responses to PEN_Q5
    aggregated_responses_q5 = {
        "Would consider (net)": [
            "Definitely would consider",
            "Probably would consider",
        ],
        "Wouldn't consider (net)": [
            "Definitely wouldn't consider",
            "Probably would consider",
        ],
    }

    q5_proportions_with_ci = modify_proportion_crosstab_with_ci(
        base_data=base_data,
        proportions_df=q5_tables["proportions"],
        aggregated_responses_column_dict=aggregated_responses_q5,
        aggregated_responses_index_dict={},
        index_survey_code=index_survey_code,
    ).drop(columns="Total")

    return q5_tables, q5_proportions_with_ci
