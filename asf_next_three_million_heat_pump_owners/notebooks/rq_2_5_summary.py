# -*- coding: utf-8 -*-
# ---
# title: "RQs 2 & 5. Prior awareness and exposure"
# format:
#   revealjs:
#     scrollable: true
#     smaller: true
#     embed-resources: true
#     css: styles.css
#     slide-number: true
#     footer: "Identifying the next three million heat pump owners: YouGov Survey (RQs 2 & 5. Prior awareness and exposure)"
# execute:
#   enabled: true
#   echo: false
#   output: true
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: asf_next_three_million_heat_pump_owners
#     language: python
#     name: python3
# ---

# %%
## Instructions for rendering
# 1. Save as a separate .ipynb file
# 2. Run: quarto render asf_next_three_million_heat_pump_owners/notebooks/[FILE_NAME].ipynb --to revealjs

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, chi2_contingency

from asf_next_three_million_heat_pump_owners import config
from asf_next_three_million_heat_pump_owners.utils import summary
from asf_next_three_million_heat_pump_owners.getters import load_data

# %%
# Load cleaned data
data, wales_data = load_data.get_analysis_ready_data()

# Get code to full question look-up dictionary
code_question_lookup = config.get("code_question_lookup")

# %%
# Load cleaned data
cleaned_data_path = f"""/mnt/g/Shared drives/A Sustainable Future/1. Reducing household emissions/2. Projects Research Work/51.Identifying the next 3 million heat pump owners/YouGov survey data/Analysis ready data/20250409_yougov_survey_clean_data.parquet"""
data = pd.read_parquet(cleaned_data_path, engine="pyarrow")
code_question_lookup = config.get("code_question_lookup")

cleaned_wales_data_path = f"""/mnt/g/Shared drives/A Sustainable Future/1. Reducing household emissions/2. Projects Research Work/51.Identifying the next 3 million heat pump owners/YouGov survey data/Analysis ready data/20250514_yougov_survey_clean_data_wales_boost.parquet"""
wales_data = pd.read_parquet(cleaned_wales_data_path)

# %% [markdown]
# # RQ 2. Before taking this survey, had UK householders ever heard of heat pumps as a home heating system?

# %% [markdown]
# ## Question that respondents were asked
#
# Before taking this survey, had you ever heard of heat pumps as a home heating system?
# (Please select the option that best applies)

# %% [markdown]
# ## Number of respondents
# - Respondents who said they had a heat pump were not asked (n = 135).
# - From recoding free text responses, however, we identified three responses that indicated that the respondent has a heat pump. As these respondents should not have been asked, we have excluded them from the question base.
# - Base (unweighted): 6,890
# - Base (unweighted), after recoding: 6,887
# - Base (weighted), after recoding: 6,875

# %%
q4_base_data = data[~(data["PEN_Q2_3_recoded"] == True)]

# %%
# n = 16 excluded from Wales dataset who have a heat pump
q4_base_data_wales = wales_data[~(wales_data["PEN_Q2_3_recoded"] == True)]

# %% [markdown]
# ## RQ 2. Overall

# %% [markdown]
# Most people had heard of heat pumps (89%), comprising of 55% who know what they are and 34% who don't.

# %%
# Summary dataframe
q4_uk = summary.create_single_question_summary_frame(q4_base_data, "PEN_Q4")

# Rename columns
q4_uk.columns = ["Count", "Weighted count", "Percentage", "Weighted percentage"]

# Drop redundant rows
q4_uk = q4_uk.drop(index="Not asked").dropna().round(1)

# Add confidence intervals for weighted proportions (95% ci)
weighted_pct_ci = summary.weighted_props_ci(q4_base_data, "PEN_Q4", "weight") * 100
q4_uk = q4_uk.join(weighted_pct_ci[["Lower CI", "Upper CI", "SE"]])

# Display only select columns
q4_uk = q4_uk.round(1)
q4_uk[
    ["Weighted count", "Weighted percentage", "Lower CI", "Upper CI", "SE"]
].style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).format(
    {
        "Weighted count": "{:.0f}",
        "Weighted percentage": "{:.1f}",
        "Lower CI": "{:.1f}",
        "Upper CI": "{:.1f}",
        "SE": "{:.1f}",
    }
).set_caption(
    "Before taking this survey, had you ever heard of heat pumps as a home heating system?"
)

# %% [markdown]
# ## RQ 2. By nation

# %% [markdown]
# The proportions of responses are similar across E/S/W.

# %%
# Individual nation datasets
q4_base_data_england = q4_base_data[q4_base_data["gornewUK_6"] == True]
# q4_base_data_wales = q4_base_data[q4_base_data["gornewUK_7"] == True]
q4_base_data_scotland = q4_base_data[q4_base_data["gornewUK_8"] == True]
q4_base_data_ni = q4_base_data[q4_base_data["gornewUK_9"] == True]

# Dataset lookup dictionary
q4_base_data_lookup = {
    "uk": q4_base_data,
    "england": q4_base_data_england,
    "wales": q4_base_data_wales,
    "scotland": q4_base_data_scotland,
    "ni": q4_base_data_ni,
}

# Create comparison table for all nations and UK
q4_nations_df, q4_nations_pct_df = summary.create_nations_summary_tables(
    q4_base_data, q4_base_data_wales, "PEN_Q4", True
)
q4_nations_pct_df = q4_nations_pct_df.drop(["Northern Ireland", "UK"])

# %%
summary.create_stacked_horizontal_bar_chart(
    q4_nations_pct_df, "Percentage", "Nation", 10, (10, 5)
).show()

# %%
# Compile 95% ci tables for each nation
weighted_props_ci_nations = {}
for nation, nation_data in q4_base_data_lookup.items():
    weighted_props_ci_nations[nation] = (
        summary.weighted_props_ci(nation_data, "PEN_Q4", "weight") * 100
    ).round(1)

# Modify original nation comparison table with lower and upper CI values
q4_nations_pct_df_ci = q4_nations_pct_df.copy(deep=True)

q4_nations_pct_df_ci = q4_nations_pct_df_ci.astype(str)

for index in q4_nations_pct_df_ci.index:
    for response in q4_nations_pct_df_ci.columns:
        lower = weighted_props_ci_nations[index.lower()].loc[response, "Lower CI"]
        upper = weighted_props_ci_nations[index.lower()].loc[response, "Upper CI"]
        q4_nations_pct_df_ci.loc[index, response] = (
            f"""{q4_nations_pct_df_ci.loc[index, response]} ({lower} - {upper})"""
        )

# Add sample size
q4_nations_pct_df_ci = (
    q4_nations_pct_df_ci.join(q4_nations_df["Nation total"])
    .assign(Nation_total=lambda df: df["Nation total"].round(0).astype(int))
    .rename(columns={"Nation total": "n"})
)

# Format display
q4_nations_pct_df_ci.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) responses to 'Before taking this survey, had you ever heard of heat pumps as a home heating system?'"
)

# %% [markdown]
# ## RQ 2. By tenure

# %% [markdown]
# Most people of all tenure types have heard of heat pumps, with homeowners having the highest proportion (95%).
#
# Homeowners also have the greatest proportion of respondents who know what heat pumps are (66%), compared to renters (39%) and others (38%).

# %%
# Create Q4 response breakdown by tenure table for each nation

# Define response aggregation structure
q4_response_aggregation_dict = {
    "Have heard (net)": [
        "I have heard of heat pumps, and I know what they are",
        "I have heard of heat pumps, but I don't know what they are",
    ],
}

q4_tenure_tables = {}
for group, base_data in q4_base_data_lookup.items():
    tables = {}
    counts_by_tenure, proportions_by_tenure = summary.generate_tenure_breakdown(
        base_data, "PEN_Q4", q4_response_aggregation_dict, proportions_axis=1
    )
    tables["counts"] = counts_by_tenure
    tables["proportions"] = proportions_by_tenure

    q4_tenure_tables[group] = tables

# %%
# Plot broad tenure categories comparison
q4_tenure_pct_df = (
    q4_tenure_tables["uk"]["proportions"]
    .reindex(index=["Own (net)", "Rent (net)", "Neither (net)", "Other"])
    .drop(columns=["Total", "Have heard (net)"])
)
summary.create_stacked_horizontal_bar_chart(
    q4_tenure_pct_df, "Percentage", "Tenure", 10, (10, 5)
).show()

# %%
# 95% confidence intervals for proportions
# Aggregated responses to profile_house_tenure
aggregated_responses_tenure = {
    "Own (net)": [
        "Own - outright",
        "Own - with a mortgage",
        "Own (part-own) - through shared ownership scheme (i.e. pay part mortgage, part rent)",
    ],
    "Rent (net)": [
        "Rent - from a private landlord",
        "Rent - from my local authority",
        "Rent - from a housing association",
    ],
    "Neither (net)": [
        "Neither - I live with my parents, family or friends but pay some rent to them",
        "Neither - I live rent-free with my parents, family or friends",
    ],
}

q4_tenure_proportions_with_ci = summary.modify_proportion_crosstab_with_ci(
    base_data=q4_base_data,
    proportions_df=q4_tenure_tables["uk"]["proportions"],
    aggregated_responses_column_dict=q4_response_aggregation_dict,
    aggregated_responses_index_dict=aggregated_responses_tenure,
    index_survey_code="profile_house_tenure",
).drop(columns="Total")

# %%
# Display broad tenure categories for UK, proportions with 95% CI
q4_tenure_proportions_with_ci_summary = (
    q4_tenure_proportions_with_ci.join(q4_tenure_tables["uk"]["counts"]["Total"])
    .assign(Total=lambda df: df["Total"].round(0).astype(int))
    .rename(columns={"Total": "n"})
)

q4_tenure_proportions_with_ci_summary.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) responses to 'Before taking this survey, had you ever heard of heat pumps as a home heating system?'"
)

# %%
# Chi-squared contingency test across tenure
q4_tenure_counts_summary = q4_tenure_tables["uk"]["counts"].drop(
    index=["Own (net)", "Rent (net)", "Neither (net)", "Total"],
    columns=["Total", "Have heard (net)"],
)
chi2_stat, p_val, dof, expected = chi2_contingency(q4_tenure_counts_summary)

if p_val < 0.05:
    print(
        f"A chi-squared test of independence found a significant association between tenure subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
else:
    print(
        f"A chi-squared test of independence showed no significant relationship between tenure subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )

# %% [markdown]
# ## RQ 2. By household income

# %% [markdown]
# Most people in all income brackets have heard of heat pumps, but the proportion is generally greater in higher income brackets. The income bracket with the largest share of people who have heard of heat pumps is £100-150k. The income bracket with the largest proportion who have never heard of heat pumps is <£10k.

# %%
# Create Q5 response breakdown by tenure table for each nation
q4_income_tables = {}
for group, base_data in q4_base_data_lookup.items():
    tables = {}
    counts_by_income, proportions_by_income = summary.generate_income_breakdown(
        base_data,
        "PEN_Q4",
        q4_response_aggregation_dict,
        proportions_axis=1,
        aggregate_income=True,
    )
    tables["counts"] = counts_by_income
    tables["proportions"] = proportions_by_income

    q4_income_tables[group] = tables

# %%
# Plot income bracket comparison table
q4_income_pct_df = (
    q4_income_tables["uk"]["proportions"]
    .reindex(
        index=[
            "under £10,000 per year (net)",
            "£10,000 to £29,999 per year (net)",
            "£30,000 to £49,999 per year (net)",
            "£50,000 to £69,999 per year (net)",
            "£70,000 to £99,999 per year",
            "£100,000 to £149,999 per year",
            "£150,000 and over",
            "Don't know",
            "Prefer not to answer",
        ]
    )
    .drop(columns=["Total", "Have heard (net)"])
)
summary.create_stacked_horizontal_bar_chart(
    q4_income_pct_df, "Percentage", "Income bracket", 10, (10, 5)
).show()

# %%
# 95% confidence intervals for proportions

# Aggregated responses to profile_gross_household
aggregated_responses_income = {
    "under £10,000 per year (net)": [
        "under £5,000 per year",
        "£5,000 to £9,999 per year",
    ],
    "£10,000 to £29,999 per year (net)": [
        "£10,000 to £14,999 per year",
        "£15,000 to £19,999 per year",
        "£20,000 to £24,999 per year",
        "£25,000 to £29,999 per year",
    ],
    "£30,000 to £49,999 per year (net)": [
        "£30,000 to £34,999 per year",
        "£35,000 to £39,999 per year",
        "£40,000 to £44,999 per year",
        "£45,000 to £49,999 per year",
    ],
    "£50,000 to £69,999 per year (net)": [
        "£50,000 to £59,999 per year",
        "£60,000 to £69,999 per year",
    ],
}

q4_income_proportions_with_ci = summary.modify_proportion_crosstab_with_ci(
    base_data=q4_base_data,
    proportions_df=q4_income_tables["uk"]["proportions"],
    aggregated_responses_column_dict=q4_response_aggregation_dict,
    aggregated_responses_index_dict=aggregated_responses_income,
    index_survey_code="profile_gross_household",
).drop(columns="Total")

# %%
# Show only aggregated income brackets
q4_income_proportions_with_ci_summary = (
    q4_income_proportions_with_ci.reindex(
        [
            "under £10,000 per year (net)",
            "£10,000 to £29,999 per year (net)",
            "£30,000 to £49,999 per year (net)",
            "£50,000 to £69,999 per year (net)",
            "£70,000 to £99,999 per year",
            "£100,000 to £149,999 per year",
            "£150,000 and over",
            "Don't know",
            "Prefer not to answer",
        ]
    )
    .join(q4_income_tables["uk"]["counts"]["Total"].round(0))
    .rename(columns={"Total": "n"})
    .assign(n=lambda df: df["n"].astype(int))
)

q4_income_proportions_with_ci_summary.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) responses to 'Before taking this survey, had you ever heard of heat pumps as a home heating system?'"
)

# %%
# Chi-squared contingency test across income brackets (net)
q4_income_counts_summary = (
    q4_income_tables["uk"]["counts"]
    .reindex(
        index=[
            "under £10,000 per year (net)",
            "£10,000 to £29,999 per year (net)",
            "£30,000 to £49,999 per year (net)",
            "£50,000 to £69,999 per year (net)",
            "£70,000 to £99,999 per year",
            "£100,000 to £149,999 per year",
            "£150,000 and over",
        ]
    )
    .drop(columns=["Total", "Have heard (net)"])
)
chi2_stat, p_val, dof, expected = chi2_contingency(q4_income_counts_summary)

if p_val < 0.05:
    print(
        f"A chi-squared test of independence found a significant association between income group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
else:
    print(
        f"A chi-squared test of independence showed no significant relationship between income group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )

# %% [markdown]
# ## RQ 2. By age

# %% [markdown]
# The proportion of people who have heard of heat pumps increases with increasing age group. For instance, 75% of 18-24 year olds vs. 96% of 55+ year olds.

# %%
# Create Q5 response breakdown by age table for the UK
q4_age_tables = {}

# Counts
q4_age_tables["counts"] = pd.crosstab(
    index=q4_base_data["profile_julesage"],
    columns=q4_base_data["PEN_Q4"],
    values=q4_base_data["weight"],
    aggfunc="sum",
    margins=True,
    margins_name="Total",
)

# Proportions
q4_age_tables["proportions"] = pd.crosstab(
    index=q4_base_data["profile_julesage"],
    columns=q4_base_data["PEN_Q4"],
    values=q4_base_data["weight"],
    aggfunc="sum",
    normalize="index",
)
q4_age_tables["proportions"]["Total"] = q4_age_tables["proportions"].sum(axis=1)
q4_age_tables["proportions"] = (q4_age_tables["proportions"] * 100).round(1)

# Add aggregated responses
for df in q4_age_tables.values():
    df["Have heard (net)"] = (
        df["I have heard of heat pumps, and I know what they are"]
        + df["I have heard of heat pumps, but I don't know what they are"]
    )
    df.drop(columns="Not asked", inplace=True)

# %%
summary.create_stacked_horizontal_bar_chart(
    q4_age_tables["proportions"].drop(columns=["Have heard (net)", "Total"]),
    "Percentage",
    "Age group",
    10,
    (10, 5),
).show()

# %%
# 95% confidence intervals for proportions
q4_age_proportions_with_ci = summary.modify_proportion_crosstab_with_ci(
    base_data=q4_base_data,
    proportions_df=q4_age_tables["proportions"],
    aggregated_responses_column_dict=q4_response_aggregation_dict,
    aggregated_responses_index_dict={},
    index_survey_code="profile_julesage",
).drop(columns="Total")

q4_age_proportions_with_ci.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) responses to 'Before taking this survey, had you ever heard of heat pumps as a home heating system?'"
)

# %%
# Chi-squared contingency test across age groups
q4_age_counts_summary = q4_age_tables["counts"].drop(
    columns=["Total", "Have heard (net)"],
    index="Total",
)
chi2_stat, p_val, dof, expected = chi2_contingency(q4_age_counts_summary)

if p_val < 0.05:
    print(
        f"A chi-squared test of independence found a significant association between age group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
else:
    print(
        f"A chi-squared test of independence showed no significant relationship between age group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )

# %%
## RQ 2. Whether home has energy tech, space/infrastructure or boiler replacement opportunity

# %%
# # Create combined counts and proportion tables
# q3_options = [
#     option
#     for option in code_question_lookup.keys()
#     if "Q3_" in option and "collapsed" not in option
# ]

# q3_q4_option_tables = {}
# for option in q3_options:
#     counts = pd.crosstab(
#         index=q4_base_data[option],
#         columns=q4_base_data["PEN_Q4"],
#         values=q4_base_data["weight"],
#         aggfunc="sum",
#         margins=True,
#         margins_name="Total",
#     ).drop(columns="Not asked")
#     proportions = (
#         pd.crosstab(
#             index=q4_base_data[option],
#             columns=q4_base_data["PEN_Q4"],
#             values=q4_base_data["weight"],
#             aggfunc="sum",
#             margins=True,
#             normalize="index",
#         ).drop(columns="Not asked")
#         * 100
#     ).round(1)
#     q3_q4_option_tables[option] = {"counts": counts, "proportions": proportions}

# q3_q4_counts = pd.DataFrame()
# q3_q4_proportions = pd.DataFrame()
# for option in q3_options:
#     q3_q4_counts = pd.concat(
#         [q3_q4_counts, q3_q4_option_tables[option]["counts"]], axis=0
#     ).drop(index=[False, "Total"])
#     q3_q4_proportions = pd.concat(
#         [q3_q4_proportions, q3_q4_option_tables[option]["proportions"]], axis=0
#     ).drop(index=[False, "All"])

# option_names = []
# for option in q3_options:
#     option_names.append(f"{code_question_lookup.get(option).split(':')[1]}")
# q3_q4_counts.index = option_names
# q3_q4_proportions.index = option_names

# for df in [q3_q4_counts, q3_q4_proportions]:
#     df.loc[:, "Have heard (net)"] = (
#         df.loc[:, "I have heard of heat pumps, and I know what they are"]
#         + df.loc[:, "I have heard of heat pumps, but I don't know what they are"]
#     )

# %%
# 95% confidence intervals for proportions
# To be completed

# %%
# # Display reordered summary table
# q3_q4_proportions_summary = q3_q4_proportions.sort_values(
#     by="Have heard (net)", ascending=False
# )
# q3_q4_proportions_summary = q3_q4_proportions_summary.join(
#     q3_q4_counts["Total"].astype(int)
# )
# q3_q4_proportions_summary = q3_q4_proportions_summary.rename(
#     columns={"Total": "n (% of population)"}
# )
# q3_q4_proportions_summary["n (% of population)"] = q3_q4_proportions_summary[
#     "n (% of population)"
# ].apply(lambda x: f"{x} ({(x/6875)*100:.1f}%)")
# q3_q4_proportions_summary

# %% [markdown]
# # RQ 5. What experiences have people had with heat pumps?

# %% [markdown]
# ## Question that respondents were asked

# %% [markdown]
# - PEN_Q6: Before taking this survey, have you ever seen what a heat pump looks like?
# - PEN_Q7: Have you ever been to a home that has a heat pump?
# - PEN_Q8: Does anyone in your family, close friends or neighbours own a heat pump?
# - PEN_Q9: Have you ever had a conversation with your family, close friends or neighbours about installing a heat pump in your home?

# %% [markdown]
# ## Number of respondents
# - Respondents who said they had a heat pump (n = 135) or said they had never heard of heat pumps (n = 645) were not asked.
# - From recoding free text responses, however, we identified three responses that indicated that the respondent has a heat pump. As these respondents should not have been asked, we have excluded them from the question base.
# - Base (unweighted): 6,245
# - Base (unweighted), after recoding: 6,242
# - Base (weighted), after recoding: 6,136

# %%
q6_base_data = data[
    ~(data["PEN_Q2_3_recoded"] == True)
    & ~(data["PEN_Q4"] == "I have never heard of heat pumps")
]

# %%
q6_base_data_wales = wales_data[
    ~(wales_data["PEN_Q2_3_recoded"] == True)
    & ~(wales_data["PEN_Q4"] == "I have never heard of heat pumps")
]

# %%
hp_experience_lookup = {
    "hp_experience_6": "Seen a heat pump",
    "hp_experience_7": "Been to a home with a heat pump",
    "hp_experience_8": "Family/close friends/neighbours have a heat pump",
    "hp_experience_9": "Have had a conversation about installing a heat pump",
    "hp_experience_any": "At least one experience",
}

# %%
# Convert to multiple column "select all that apply" format
q6_base_data = q6_base_data.copy()

for label in ["6", "7", "8", "9"]:
    if label == "8":
        q6_base_data[f"hp_experience_{label}"] = (
            q6_base_data[f"PEN_Q{label}"] == "Yes, they do"
        )
    else:
        q6_base_data[f"hp_experience_{label}"] = (
            q6_base_data[f"PEN_Q{label}"] == "Yes, I have"
        )
    q6_base_data[f"hp_experience_{label}"] = pd.Categorical(
        q6_base_data[f"hp_experience_{label}"], categories=[True, False]
    )

# Create a column for any experience
q6_base_data["hp_experience_any"] = q6_base_data[
    [f"hp_experience_{label}" for label in ["6", "7", "8", "9"]]
].any(axis=1)

# %%
# Repeat for Wales data
q6_base_data_wales = q6_base_data_wales.copy()

for label in ["6", "7", "8", "9"]:
    if label == "8":
        q6_base_data_wales[f"hp_experience_{label}"] = (
            q6_base_data_wales[f"PEN_Q{label}"] == "Yes, they do"
        )
    else:
        q6_base_data_wales[f"hp_experience_{label}"] = (
            q6_base_data_wales[f"PEN_Q{label}"] == "Yes, I have"
        )
    q6_base_data_wales[f"hp_experience_{label}"] = pd.Categorical(
        q6_base_data_wales[f"hp_experience_{label}"], categories=[True, False]
    )

# Create a column for any experience
q6_base_data_wales["hp_experience_any"] = q6_base_data_wales[
    [f"hp_experience_{label}" for label in ["6", "7", "8", "9"]]
].any(axis=1)

# %% [markdown]
# ## RQ 5. UK and by nation

# %% [markdown]
# **At least one experience**
#
# Overall in the UK, most people have had at least one prior experience with heat pumps (57%). This proportion is similar across E/S/W, with it being slightly higher in Wales (61%).
#
# **Seen a heat pump**
#
# Around half of people in each nation, and overall in the UK, have seen what a heat pump looks like. This proportion again is slightly higher in wales (56%).
#
# **Been to a home with a heat pump**
#
# Around 1 in 7 people in England and Wales, and overall in the UK, have been to a home with a heat pump. This is higher in Scotland where it is 1 in 5.
#
# **Family/close friends/neighbours have a heat pump**
#
# Around 1 in 10 people said they know that they have family, close friends or neighbours with a heat pump. This proportion is slightly higher in Scotland (14%).
#
# **Have had a conversation with family/close friends/neighbours about installing a heat pump**
#
# About 1 in 5 in England and UK overall have had a conversation about installing a heat pump. This proportion is higher in Scotland and Wales where it is about 1 in 4.
#
#

# %%
# Create nation summary tables
q6_base_data_england = q6_base_data[q6_base_data["gornewUK_6"] == True]
# q6_base_data_wales = q6_base_data[q6_base_data["gornewUK_7"] == True]
q6_base_data_scotland = q6_base_data[q6_base_data["gornewUK_8"] == True]
q6_base_data_ni = q6_base_data[q6_base_data["gornewUK_9"] == True]

q6_base_data_lookup = {
    "uk": q6_base_data,
    "england": q6_base_data_england,
    "wales": q6_base_data_wales,
    "scotland": q6_base_data_scotland,
    "ni": q6_base_data_ni,
}

hp_experience_tables = {}
for experience in hp_experience_lookup.keys():
    counts, proportions = summary.create_nations_summary_tables(
        q6_base_data, q6_base_data_wales, experience, False
    )
    counts = counts.drop("Northern Ireland")
    proportions = proportions.drop("Northern Ireland")
    hp_experience_tables[experience] = {"counts": counts, "proportions": proportions}

# %%
# Summarise all experiences
hp_experience_counts = pd.DataFrame()
hp_experience_proportions = pd.DataFrame()

for experience in hp_experience_lookup.keys():
    hp_experience_counts = pd.concat(
        [hp_experience_counts, hp_experience_tables[experience]["counts"][True]], axis=1
    )
    hp_experience_proportions = pd.concat(
        [
            hp_experience_proportions,
            hp_experience_tables[experience]["proportions"][True],
        ],
        axis=1,
    )

for df in [hp_experience_counts, hp_experience_proportions]:
    df.columns = [
        "Seen a heat pump",
        "Been to a home with a heat pump",
        "Family/close friends/neighbours have a heat pump",
        "Have had a conversation about installing a heat pump",
        "At least one experience",
    ]

# %%
experience_df = (
    hp_experience_proportions.drop(index="UK")
    .reset_index()
    .rename(columns={"index": "Nation"})
)
experience_df_long = experience_df.melt(
    id_vars="Nation", var_name="Experience", value_name="Percentage"
)

plt.figure(figsize=(10, 5))
ax = sns.barplot(data=experience_df_long, y="Experience", x="Percentage", hue="Nation")
ax.xaxis.grid(True, linestyle="-", alpha=0.3)
plt.xlabel("Percentage (%)")
plt.ylabel("Experience")
plt.legend(title="Nation")
plt.tight_layout()
plt.show()

# %%
# Compile 95% ci tables for each experience and each nation
hp_experience_props_ci = {}
for nation, nation_data in q6_base_data_lookup.items():
    nation_tables = {}
    for i in [6, 7, 8, 9, "any"]:
        nation_tables[f"hp_experience_{i}"] = (
            summary.weighted_props_ci(nation_data, f"hp_experience_{i}", "weight") * 100
        ).round(1)
    hp_experience_props_ci[nation] = nation_tables

# Modify original nation comparison tables with lower and upper CI values
hp_experience_proportions_with_ci = hp_experience_proportions.copy(deep=True)
hp_experience_proportions_with_ci = hp_experience_proportions_with_ci.astype(str)

for experience in hp_experience_lookup.keys():
    for index in hp_experience_proportions_with_ci.index:
        lower = hp_experience_props_ci[index.lower()][experience].loc[True, "Lower CI"]
        upper = hp_experience_props_ci[index.lower()][experience].loc[True, "Upper CI"]

        hp_experience_proportions_with_ci.loc[
            index, hp_experience_lookup[experience]
        ] = f"""{hp_experience_proportions_with_ci.loc[index, hp_experience_lookup[experience]]} ({lower}-{upper})"""

# %%
hp_experience_proportions_with_ci.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) who have had prior experience with heat pumps"
)

# %%
# Chi-squared contingency test across nations
# Each experience question

for experience in hp_experience_lookup.keys():
    if experience != "hp_experience_any":
        chi2_stat, p_val, dof, expected = chi2_contingency(
            hp_experience_tables[experience]["counts"].drop(
                index="UK", columns="Nation total"
            )
        )

        if p_val < 0.05:
            print(
                f"{experience}: A chi-squared test of independence found a significant association between nations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
            )
        else:
            print(
                f"{experience}: A chi-squared test of independence showed no significant relationship between nations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
            )

# %% [markdown]
# ## RQ 5. By tenure

# %% [markdown]
# Homeowners have the highest proportion who have had at least one prior experience with heat pumps (65%), compared to 42% of renters and 44% of others.
#
# The tenure subgroup who have the highest proportion who have had at least one prior experience is outright homeowners (70%). Conversely, people who rent from local authority have the lowest proportion (36%).

# %%
# Create experience and tenure breakdown tables (counts and proportions) for each nation
experience_tenure_tables = {}
for experience in hp_experience_lookup.keys():
    experience_tenure_tables[experience] = {}
    for group, base_data in q6_base_data_lookup.items():
        tables = {}
        counts_by_response, proportions_by_response = summary.generate_tenure_breakdown(
            base_data, experience, proportions_axis=1
        )
        tables["counts"] = counts_by_response
        tables["proportions"] = proportions_by_response

        experience_tenure_tables[experience][group] = tables

# %%
# Combine all experiences in one table
hp_experience_tenure_counts = pd.concat(
    [
        experience_tenure_tables[experience]["uk"]["counts"][True]
        for experience in hp_experience_lookup.keys()
    ],
    axis=1,
)

hp_experience_tenure_proportions = pd.concat(
    [
        experience_tenure_tables[experience]["uk"]["proportions"][True]
        for experience in hp_experience_lookup.keys()
    ],
    axis=1,
)

for df in [hp_experience_tenure_counts, hp_experience_tenure_proportions]:
    df.columns = [
        "Seen a heat pump",
        "Been to a home with a heat pump",
        "Family/close friends/neighbours have a heat pump",
        "Have had a conversation about installing a heat pump",
        "At least one experience",
    ]

# %%
experience_tenure_df = (
    hp_experience_tenure_proportions.reindex(
        index=["Own (net)", "Rent (net)", "Neither (net)", "Other"]
    )
    .reset_index()
    .rename(columns={"profile_house_tenure": "Tenure"})
)

experience_tenure_df_long = experience_tenure_df.melt(
    id_vars="Tenure", var_name="Experience", value_name="Percentage"
)

plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=experience_tenure_df_long, y="Experience", x="Percentage", hue="Tenure"
)
ax.xaxis.grid(True, linestyle="-", alpha=0.3)
plt.xlabel("Percentage of tenure subpopulation (%)")
plt.ylabel("Experience")
plt.legend(title="Tenure")
plt.tight_layout()
plt.show()

# %%
# Compile 95% ci tables for each experience
experience_tenure_tables_with_ci = {}
for experience in hp_experience_lookup.keys():
    experience_tenure_tables[experience]["uk"]["proportions"].columns = (
        experience_tenure_tables[experience]["uk"]["proportions"].columns.astype(str)
    )
    df = summary.modify_proportion_crosstab_with_ci(
        q6_base_data,
        experience_tenure_tables[experience]["uk"]["proportions"],
        aggregated_responses_column_dict={},
        aggregated_responses_index_dict=aggregated_responses_tenure,
        index_survey_code="profile_house_tenure",
    )
    experience_tenure_tables_with_ci[experience] = df

# Combine all experiences in one table
hp_experience_tenure_proportions_with_ci = pd.concat(
    [
        experience_tenure_tables_with_ci[experience]["True"]
        for experience in hp_experience_lookup.keys()
    ],
    axis=1,
)
hp_experience_tenure_proportions_with_ci.columns = [
    "Seen a heat pump",
    "Been to a home with a heat pump",
    "Family/close friends/neighbours have a heat pump",
    "Have had a conversation about installing a heat pump",
    "At least one experience",
]

hp_experience_tenure_proportions_with_ci.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) who have had prior experience with heat pumps"
)

# %%
# Chi-squared contingency test across tenure groups
# Each experience question

for experience in hp_experience_lookup.keys():
    if experience != "hp_experience_any":
        chi2_stat, p_val, dof, expected = chi2_contingency(
            experience_tenure_tables[experience]["uk"]["counts"].drop(
                index=["Own (net)", "Rent (net)", "Neither (net)", "Other"],
                columns="Total",
            )
        )

        if p_val < 0.05:
            print(
                f"{experience}: A chi-squared test of independence found a significant association between tenure group and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
            )
        else:
            print(
                f"{experience}: A chi-squared test of independence showed no significant relationship between tenure group and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
            )

# %% [markdown]
# ## RQ 5. By income

# %% [markdown]
# Higher income brackets have higher proportion of those with prior experience with heat pumps, across all experience types. For example, 73% of people with household income over £150k have had at least one prior experience compared to 40% of those who earn under <£10k.

# %%
# Create experience and tenure breakdown tables (counts and proportions) for each nation
experience_income_tables = {}
for experience in hp_experience_lookup.keys():
    experience_income_tables[experience] = {}
    for group, base_data in q6_base_data_lookup.items():
        tables = {}
        counts_by_response, proportions_by_response = summary.generate_income_breakdown(
            base_data, experience, proportions_axis=1
        )
        tables["counts"] = counts_by_response
        tables["proportions"] = proportions_by_response

        experience_income_tables[experience][group] = tables

# %%
# Combine all experiences in one table
hp_experience_income_counts = pd.concat(
    [
        experience_income_tables[experience]["uk"]["counts"][True]
        for experience in hp_experience_lookup.keys()
    ],
    axis=1,
)

hp_experience_income_proportions = pd.concat(
    [
        experience_income_tables[experience]["uk"]["proportions"][True]
        for experience in hp_experience_lookup.keys()
    ],
    axis=1,
)

for df in [hp_experience_income_counts, hp_experience_income_proportions]:
    df.columns = [
        "Seen a heat pump",
        "Been to a home with a heat pump",
        "Family/close friends/neighbours have a heat pump",
        "Have had a conversation about installing a heat pump",
        "At least one experience",
    ]

# %%
experience_income_df = (
    hp_experience_income_proportions.reindex(
        index=[
            "under £10,000 per year (net)",
            "£10,000 to £29,999 per year (net)",
            "£30,000 to £49,999 per year (net)",
            "£50,000 to £69,999 per year (net)",
            "£70,000 to £99,999 per year",
            "£100,000 to £149,999 per year",
            "£150,000 and over",
            # "Don't know",
            # "Prefer not to answer",
        ]
    )
    .reset_index()
    .rename(columns={"profile_gross_household": "Income"})
)
experience_income_df_long = experience_income_df.melt(
    id_vars="Income", var_name="Experience", value_name="Percentage"
)

plt.figure(figsize=(9, 9))
ax = sns.barplot(
    data=experience_income_df_long, y="Experience", x="Percentage", hue="Income"
)
ax.xaxis.grid(True, linestyle="-", alpha=0.3)
plt.xlabel("Percentage of income subpopulation (%)")
plt.ylabel("Experience")
plt.legend(title="Income")
plt.tight_layout()
plt.show()

# %%
# Compile 95% ci tables for each experience

experience_income_tables_with_ci = {}
for experience in hp_experience_lookup.keys():
    experience_income_tables[experience]["uk"]["proportions"].columns = (
        experience_income_tables[experience]["uk"]["proportions"].columns.astype(str)
    )
    df = summary.modify_proportion_crosstab_with_ci(
        q6_base_data,
        experience_income_tables[experience]["uk"]["proportions"],
        aggregated_responses_column_dict={},
        aggregated_responses_index_dict=aggregated_responses_income,
        index_survey_code="profile_gross_household",
    )
    experience_income_tables_with_ci[experience] = df

# Combine all experiences in one table
hp_experience_income_proportions_with_ci = pd.concat(
    [
        experience_income_tables_with_ci[experience]["True"]
        for experience in hp_experience_lookup.keys()
    ],
    axis=1,
)
hp_experience_income_proportions_with_ci.columns = [
    "Seen a heat pump",
    "Been to a home with a heat pump",
    "Family/close friends/neighbours have a heat pump",
    "Have had a conversation about installing a heat pump",
    "At least one experience",
]

hp_experience_income_proportions_with_ci.reindex(
    index=[
        "under £10,000 per year (net)",
        "£10,000 to £29,999 per year (net)",
        "£30,000 to £49,999 per year (net)",
        "£50,000 to £69,999 per year (net)",
        "£70,000 to £99,999 per year",
        "£100,000 to £149,999 per year",
        "£150,000 and over",
        "Don't know",
        "Prefer not to answer",
    ]
).style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) who have had prior experience with heat pumps"
)

# %%
# Chi-squared contingency test across income groups
# Each experience question

for experience in hp_experience_lookup.keys():
    if experience != "hp_experience_any":
        chi2_stat, p_val, dof, expected = chi2_contingency(
            experience_income_tables[experience]["uk"]["counts"].drop(
                index=[
                    "under £10,000 per year (net)",
                    "£10,000 to £29,999 per year (net)",
                    "£30,000 to £49,999 per year (net)",
                    "£50,000 to £69,999 per year (net)",
                ],
                columns="Total",
            )
        )

        if p_val < 0.05:
            print(
                f"{experience}: A chi-squared test of independence found a significant association between income group and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
            )
        else:
            print(
                f"{experience}: A chi-squared test of independence showed no significant relationship between income group and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
            )

# %% [markdown]
# ## RQ 5. By age

# %% [markdown]
# - Any experience

# %%
# Create any experience response breakdown by age table for the UK
hp_experience_age_tables = {}

# Counts
hp_experience_age_tables["counts"] = pd.crosstab(
    index=q6_base_data["profile_julesage"],
    columns=q6_base_data["hp_experience_any"],
    values=q6_base_data["weight"],
    aggfunc="sum",
    margins=True,
    margins_name="Total",
)

# Proportions
hp_experience_age_tables["proportions"] = pd.crosstab(
    index=q6_base_data["profile_julesage"],
    columns=q6_base_data["hp_experience_any"],
    values=q6_base_data["weight"],
    aggfunc="sum",
    normalize="index",
)
hp_experience_age_tables["proportions"]["Total"] = hp_experience_age_tables[
    "proportions"
].sum(axis=1)
hp_experience_age_tables["proportions"] = (
    hp_experience_age_tables["proportions"] * 100
).round(1)

# %%
hp_experience_age_tables["proportions"]

# %% [markdown]
# - PEN_Q6: Before taking this survey, have you ever seen what a heat pump looks like?

# %%
# Create Q6 response breakdown by age table for the UK
q6_age_tables = {}

# Counts
q6_age_tables["counts"] = pd.crosstab(
    index=q6_base_data["profile_julesage"],
    columns=q6_base_data["PEN_Q6"],
    values=q6_base_data["weight"],
    aggfunc="sum",
    margins=True,
    margins_name="Total",
)

# Proportions
q6_age_tables["proportions"] = pd.crosstab(
    index=q6_base_data["profile_julesage"],
    columns=q6_base_data["PEN_Q6"],
    values=q6_base_data["weight"],
    aggfunc="sum",
    normalize="index",
)
q6_age_tables["proportions"]["Total"] = q6_age_tables["proportions"].sum(axis=1)
q6_age_tables["proportions"] = (q6_age_tables["proportions"] * 100).round(1)

# %%
q6_age_tables["proportions"]

# %% [markdown]
# - PEN_Q7: Have you ever been to a home that has a heat pump?

# %%
# Create Q7 response breakdown by age table for the UK
q7_age_tables = {}

# Counts
q7_age_tables["counts"] = pd.crosstab(
    index=q6_base_data["profile_julesage"],
    columns=q6_base_data["PEN_Q7"],
    values=q6_base_data["weight"],
    aggfunc="sum",
    margins=True,
    margins_name="Total",
)

# Proportions
q7_age_tables["proportions"] = pd.crosstab(
    index=q6_base_data["profile_julesage"],
    columns=q6_base_data["PEN_Q7"],
    values=q6_base_data["weight"],
    aggfunc="sum",
    normalize="index",
)
q7_age_tables["proportions"]["Total"] = q7_age_tables["proportions"].sum(axis=1)
q7_age_tables["proportions"] = (q7_age_tables["proportions"] * 100).round(1)

# %%
q7_age_tables["proportions"]

# %% [markdown]
# - PEN_Q8: Does anyone in your family, close friends or neighbours own a heat pump?

# %%
# Create Q8 response breakdown by age table for the UK
q8_age_tables = {}

# Counts
q8_age_tables["counts"] = pd.crosstab(
    index=q6_base_data["profile_julesage"],
    columns=q6_base_data["PEN_Q8"],
    values=q6_base_data["weight"],
    aggfunc="sum",
    margins=True,
    margins_name="Total",
)

# Proportions
q8_age_tables["proportions"] = pd.crosstab(
    index=q6_base_data["profile_julesage"],
    columns=q6_base_data["PEN_Q8"],
    values=q6_base_data["weight"],
    aggfunc="sum",
    normalize="index",
)
q8_age_tables["proportions"]["Total"] = q8_age_tables["proportions"].sum(axis=1)
q8_age_tables["proportions"] = (q8_age_tables["proportions"] * 100).round(1)

# %%
q8_age_tables["proportions"]

# %% [markdown]
# - PEN_Q9: Have you ever had a conversation with your family, close friends or neighbours about installing a heat pump in your home?

# %%
# Create Q9 response breakdown by age table for the UK
q9_age_tables = {}

# Counts
q9_age_tables["counts"] = pd.crosstab(
    index=q6_base_data["profile_julesage"],
    columns=q6_base_data["PEN_Q9"],
    values=q6_base_data["weight"],
    aggfunc="sum",
    margins=True,
    margins_name="Total",
)

# Proportions
q9_age_tables["proportions"] = pd.crosstab(
    index=q6_base_data["profile_julesage"],
    columns=q6_base_data["PEN_Q9"],
    values=q6_base_data["weight"],
    aggfunc="sum",
    normalize="index",
)
q9_age_tables["proportions"]["Total"] = q9_age_tables["proportions"].sum(axis=1)
q9_age_tables["proportions"] = (q9_age_tables["proportions"] * 100).round(1)

# %%
q9_age_tables["proportions"]

# %%
