# -*- coding: utf-8 -*-
# ---
# title: "RQ 6. How desirable is buying a house that has a heat pump be for UK households?"
# format:
#   revealjs:
#     scrollable: true
#     smaller: true
#     embed-resources: true
#     css: styles.css
#     slide-number: true
#     footer: "RQ 6. How desirable is buying a house that has a heat pump be for UK households?"
#     center-title-slide: false
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

# %% [markdown]
# ## Question that respondents were asked
#
# How desirable, if at all, would buying a house that has a heat pump be for you?

# %% [markdown]
# **Number of respondents**
# - All respondents were asked.
# - Base: 7,025

# %%
base_data = data.copy(deep=True)

# %%
base_data_wales = wales_data.copy(deep=True)

# %% [markdown]
# ## RQ 6. Overall

# %% [markdown]
# A greater proportion of respondents said a house with a heat pump would not be desirable (41%) rather than desirable (36%). Almost a quarter (23%) said that they didn't know.

# %%
# Response aggregation dict
q5d_response_aggregation_dict = {
    "Desirable (net)": ["Very desirable", "Somewhat desirable"],
    "Not desirable (net)": ["Not at all desirable", "Not very desirable"],
}

# %%
# Summary dataframe
q5d_uk = summary.create_single_question_summary_frame(base_data, "PEN_Q5Dnew")

# Rename columns
q5d_uk.columns = ["Count", "Weighted count", "Percentage", "Weighted percentage"]

# Combine categories
for combined_col, source_cols in q5d_response_aggregation_dict.items():
    q5d_uk.loc[combined_col, :] = q5d_uk.loc[source_cols, :].sum()

# Add confidence intervals for weighted proportions (95% ci)
weighted_pct_ci = summary.weighted_props_ci(base_data, "PEN_Q5Dnew", "weight") * 100
q5d_uk = q5d_uk.join(weighted_pct_ci[["Lower CI", "Upper CI", "SE"]])

# Add confidence intervals for combined weighted proportions
desirable_net = summary.weighted_props_ci_combine_categories(
    q5d_uk,
    "Somewhat desirable",
    "Very desirable",
    "Desirable (net)",
)
q5d_uk.loc["Desirable (net)", "Lower CI"] = desirable_net["Lower CI"]
q5d_uk.loc["Desirable (net)", "Upper CI"] = desirable_net["Upper CI"]
q5d_uk.loc["Desirable (net)", "SE"] = desirable_net["SE"]

not_desirable_net = summary.weighted_props_ci_combine_categories(
    q5d_uk,
    "Not very desirable",
    "Not at all desirable",
    "Not desirable (net)",
)
q5d_uk.loc["Not desirable (net)", "Lower CI"] = not_desirable_net["Lower CI"]
q5d_uk.loc["Not desirable (net)", "Upper CI"] = not_desirable_net["Upper CI"]
q5d_uk.loc["Not desirable (net)", "SE"] = not_desirable_net["SE"]

# Display only select columns
q5d_uk = q5d_uk.round(1)
q5d_uk[
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
    "How desirable, if at all, would buying a house that has a heat pump be for you?"
)

# %% [markdown]
# ## RQ 6. By nation

# %% [markdown]
# The proportion breakdown of responses is similar across England, Scotland and Wales.

# %%
# Individual nation datasets
base_data_england = base_data[base_data["gornewUK_6"] == True]
# base_data_wales = base_data[base_data["gornewUK_7"] == True]
base_data_scotland = base_data[base_data["gornewUK_8"] == True]
base_data_ni = base_data[base_data["gornewUK_9"] == True]

# Dataset lookup dictionary
base_data_lookup = {
    "uk": base_data,
    "england": base_data_england,
    "wales": base_data_wales,
    "scotland": base_data_scotland,
    "ni": base_data_ni,
}

# Create comparison table for all nations and UK
q5d_nations_df, q5d_nations_pct_df = summary.create_nations_summary_tables(
    base_data, base_data_wales, "PEN_Q5Dnew", False
)
q5d_nations_pct_df = q5d_nations_pct_df.drop("Northern Ireland")

# %%
summary.create_stacked_horizontal_bar_chart(
    q5d_nations_pct_df, "Percentage", "Nation", 10, (10, 5)
).show()

# %%
# Compile 95% ci tables for each nation
weighted_props_ci_nations = {}
for nation, nation_data in base_data_lookup.items():
    weighted_props_ci_nations[nation] = (
        summary.weighted_props_ci(nation_data, "PEN_Q5Dnew", "weight") * 100
    ).round(1)

# Modify original nation comparison table with lower and upper CI values
q5d_nations_pct_df_ci = q5d_nations_pct_df.copy(deep=True)
q5d_nations_pct_df_ci = q5d_nations_pct_df_ci.astype(str)

for index in q5d_nations_pct_df_ci.index:
    for response in q5d_nations_pct_df_ci.columns:
        lower = weighted_props_ci_nations[index.lower()].loc[response, "Lower CI"]
        upper = weighted_props_ci_nations[index.lower()].loc[response, "Upper CI"]
        q5d_nations_pct_df_ci.loc[index, response] = (
            f"""{q5d_nations_pct_df_ci.loc[index, response]} ({lower} - {upper})"""
        )

q5d_nations_pct_df_ci = (
    q5d_nations_pct_df_ci.join(q5d_nations_df["Nation total"])
    .assign(**{"Nation total": lambda df: df["Nation total"].round(0).astype(int)})
    .rename(columns={"Nation total": "n"})
)

q5d_nations_pct_df_ci.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "How desirable, if at all, would buying a house that has a heat pump be for you?"
)

# %% [markdown]
# ## RQ 6. By tenure

# %% [markdown]
# People who live with family/friends rent-free have the highest proportion who said that buying a house with a heat pump would be desirable (48%). Those renting from their local authority have the lowest (28%).
#
# People who own their homes outright have the highest proportion of people who said it would not be desirable (48%, just under half).
#
# For the renters (net) tenure group, the proportion of people who said desirable vs. not desirable are roughly the same (36% vs. 38%). For homeowners (net), however, the proportion who said not desirable (45%) is greater than that who said desirable (34%). The opposite is true for the neither/other group where more people said desirable (43%) than undesirable (27%).

# %%
# Create Q5D response breakdown by tenure table for each nation
q5d_tenure_tables = {}
for group, base_data_nation in base_data_lookup.items():
    tables = {}
    counts_by_tenure, proportions_by_tenure = summary.generate_tenure_breakdown(
        base_data_nation,
        "PEN_Q5Dnew",
        q5d_response_aggregation_dict,
        proportions_axis=1,
    )
    tables["counts"] = counts_by_tenure
    tables["proportions"] = proportions_by_tenure

    q5d_tenure_tables[group] = tables

# %%
# Plot broad tenure categories comparison
q5d_tenure_pct_df = (
    q5d_tenure_tables["uk"]["proportions"]
    .reindex(index=["Own (net)", "Rent (net)", "Neither (net)", "Other"])
    .drop(columns=["Total", "Desirable (net)", "Not desirable (net)"])
)
summary.create_stacked_horizontal_bar_chart(
    q5d_tenure_pct_df, "Percentage", "Tenure", 10, (10, 5)
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

q5d_tenure_proportions_with_ci = summary.modify_proportion_crosstab_with_ci(
    base_data=base_data,
    proportions_df=q5d_tenure_tables["uk"]["proportions"],
    aggregated_responses_column_dict=q5d_response_aggregation_dict,
    aggregated_responses_index_dict=aggregated_responses_tenure,
    index_survey_code="profile_house_tenure",
).drop(columns="Total")

# %%
# Display broad tenure categories for UK, proportions with 95% CI
q5d_tenure_proportions_with_ci_summary = (
    q5d_tenure_proportions_with_ci.join(q5d_tenure_tables["uk"]["counts"]["Total"])
    .assign(Total=lambda df: df["Total"].round(0).astype(int))
    .rename(columns={"Total": "n"})
)

q5d_tenure_proportions_with_ci_summary.drop(index="Total").style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "How desirable, if at all, would buying a house that has a heat pump be for you?"
)

# %%
# Chi-squared contingency test across tenure
q5d_tenure_counts_summary = q5d_tenure_tables["uk"]["counts"].drop(
    index=["Own (net)", "Rent (net)", "Neither (net)", "Total"],
    columns=["Total", "Desirable (net)", "Not desirable (net)"],
)
chi2_stat, p_val, dof, expected = chi2_contingency(q5d_tenure_counts_summary)

if p_val < 0.05:
    print(
        f"A chi-squared test of independence found a significant association between tenure subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
else:
    print(
        f"A chi-squared test of independence showed no significant relationship between tenure subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
