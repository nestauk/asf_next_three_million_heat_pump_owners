# -*- coding: utf-8 -*-
# ---
# title: "RQ 1. How likely are UK households to consider installing a heat pump?"
# format:
#   revealjs:
#     scrollable: true
#     smaller: true
#     embed-resources: true
#     css: styles.css
#     slide-number: true
#     footer: "RQ 1. How likely are UK households to consider installing a heat pump?"
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
# ## Question that respondents were asked (PEN_Q5)
#
# For the following questions, please imagine that in the next 5 years...(i.e. from now till February 2030), you want to install a central heating system for your home/ your current heating system needs replacing…
# A heat pump can heat a home. An outside unit (see picture) takes heat from the air, concentrates it, and puts it into radiators.
# It's like a fridge in reverse. It works effectively in all weather conditions, including on cold days.
# Heat pumps run on electricity so differ to boilers, which run on fossil fuels (gas or oil).
# This means that heat pumps can run using electricity that is generated from renewable resources, such as wind or sun, which do not produce carbon emissions…
#
# **How likely, if at all, would you be to consider installing a heat pump in your home as your main heating source?**
#
# (Please select the option that best applies. Even if you are not currently a homeowner, we are still interested in your opinion.)

# %% [markdown]
# ## Overview of all respondents (including those with a heat pump)

# %%
# Recode Q5 to change responses of those that actually do have a heat pump from Not asked to Already have a heat pump
full_sample_data = data.copy(deep=True)
full_sample_data["PEN_Q5_recoded"] = full_sample_data.apply(
    lambda row: (
        "Already have a heat pump" if row["PEN_Q2_3_recoded"] == True else row["PEN_Q5"]
    ),
    axis=1,
)

# %% [markdown]
# 2% of all respondents have a heat pump. This is similar to the percentages reported in the [Which? Sustainability Tracker 2024](https://www.which.co.uk/policy-and-insight/article/whichs-annual-sustainability-report-series-2024-home-insulation-and-heating-a0S066z6SiHV#chapter-2-the-home-heating-challenge) and the most recent [DESNZ Public Attitudes Tracker](https://www.gov.uk/government/statistics/desnz-public-attitudes-tracker-winter-2024/desnz-public-attitudes-tracker-heat-and-energy-use-in-the-home-winter-2024-uk#low-carbon-heating-systems), but they report only for homeowners or owner occupiers.

# %%
q5_full_df = (
    full_sample_data.groupby("PEN_Q5_recoded", dropna=False)["weight"]
    .sum()
    .reset_index()
    .rename(columns={"weight": "weighted Count", "PEN_Q5_recoded": "Value"})
)


q5_full_df.columns = ["How likely to consider", "weighted count"]
q5_full_df["weighted percentage"] = (
    q5_full_df["weighted count"] / q5_full_df["weighted count"].sum()
) * 100

# Formatting output
q5_full_df.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).format(
    {
        "weighted count": "{:.0f}",
        "weighted percentage": "{:.1f}",
    }
).set_caption(
    "How likely, if at all, would you be to consider installing a heat pump in your home as your main heating source?"
)

# %% [markdown]
# The following analysis focuses only on respondents that do not have a heat pump, who were asked PEN_Q5.
#
# - Respondents who said they had a heat pump were not asked (n = 135).
# - From recoding free text responses, however, we identified three responses that indicated that the respondent has a heat pump. As these respondents should not have been asked, we have excluded them from the question base.
# - Base (unweighted): 6,890
# - Base (unweighted), after recoding: 6,887
# - Base (weighted), after recoding: 6,875

# %%
q5_base_data = data[~(data["PEN_Q2_3_recoded"] == True)]

# %%
# n = 16 excluded from Wales dataset who have a heat pump
q5_base_data_wales = wales_data[~(wales_data["PEN_Q2_3_recoded"] == True)]

# %% [markdown]
# ## RQ 1. Overall

# %% [markdown]
# Almost equal shares of people who don't already have a heat pump said they would (43%) and wouldn't (44%) consider installing a heat pump in their home as their main heating source, leaving 12% who said that they didn't know.

# %%
# Define response aggregation structure
q5_response_aggregation_dict = {
    "Would consider (net)": ["Probably would consider", "Definitely would consider"],
    "Wouldn't consider (net)": [
        "Probably wouldn't consider",
        "Definitely wouldn't consider",
    ],
}

# %%
# Summary dataframe
q5_uk = summary.create_single_question_summary_frame(q5_base_data, "PEN_Q5")

# Rename columns
q5_uk.columns = ["Count", "Weighted count", "Percentage", "Weighted percentage"]

# Drop redundant rows
q5_uk = q5_uk.drop(index="Not asked").dropna().round(1)

# Combine categories
for combined_col, source_cols in q5_response_aggregation_dict.items():
    q5_uk.loc[combined_col, :] = q5_uk.loc[source_cols, :].sum()

# Add confidence intervals for weighted proportions (95% ci)
weighted_pct_ci = summary.weighted_props_ci(q5_base_data, "PEN_Q5", "weight") * 100
q5_uk = q5_uk.join(weighted_pct_ci[["Lower CI", "Upper CI", "SE"]])

# Add confidence intervals for combined weighted proportions
consider_net = summary.weighted_props_ci_combine_categories(
    q5_uk,
    "Definitely would consider",
    "Probably would consider",
    "Would consider (net)",
)
q5_uk.loc["Would consider (net)", "Lower CI"] = consider_net["Lower CI"]
q5_uk.loc["Would consider (net)", "Upper CI"] = consider_net["Upper CI"]
q5_uk.loc["Would consider (net)", "SE"] = consider_net["SE"]

not_consider_net = summary.weighted_props_ci_combine_categories(
    q5_uk,
    "Definitely wouldn't consider",
    "Probably wouldn't consider",
    "Wouldn't consider (net)",
)
q5_uk.loc["Wouldn't consider (net)", "Lower CI"] = not_consider_net["Lower CI"]
q5_uk.loc["Wouldn't consider (net)", "Upper CI"] = not_consider_net["Upper CI"]
q5_uk.loc["Wouldn't consider (net)", "SE"] = not_consider_net["SE"]

# %%
# Format display
q5_uk[
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
    "How likely, if at all, would you be to consider installing a heat pump in your home as your main heating source? (With lower and upper 95% confidence interval values for proportions)"
)

# %% [markdown]
# ## RQ 1. By nation

# %% [markdown]
# The proportions of respondents who said they definitely/probably would consider are similar across E/S/W (~42%), with a slight variation where a lower proportion of people in Wales said they they would definitely consider and a higher proportion said they would probably consider.
#
# The proportions who said they definitely/probably wouldn't consider are similar across E/S/W (~45%), as well as those who said they didn't know.

# %%
# Individual nation datasets
q5_base_data_england = q5_base_data[q5_base_data["gornewUK_6"] == True]
# q5_base_data_wales = q5_base_data[q5_base_data["gornewUK_7"] == True]
q5_base_data_scotland = q5_base_data[q5_base_data["gornewUK_8"] == True]
q5_base_data_ni = q5_base_data[q5_base_data["gornewUK_9"] == True]

# Dataset lookup dictionary
q5_base_data_lookup = {
    "uk": q5_base_data,
    "england": q5_base_data_england,
    "wales": q5_base_data_wales,
    "scotland": q5_base_data_scotland,
    "ni": q5_base_data_ni,
}

# Create comparison table for all nations and UK
q5_nations_df, q5_nations_pct_df = summary.create_nations_summary_tables(
    q5_base_data, q5_base_data_wales, "PEN_Q5", True
)
q5_nations_pct_df = q5_nations_pct_df.drop(["Northern Ireland", "UK"])

# %%
summary.create_stacked_horizontal_bar_chart(
    q5_nations_pct_df, "Percentage", "Nation", 10, (10, 5)
).show()

# %%
# Compile 95% ci tables for each nation
weighted_props_ci_nations = {}
for nation, nation_data in q5_base_data_lookup.items():
    weighted_props_ci_nations[nation] = (
        summary.weighted_props_ci(nation_data, "PEN_Q5", "weight") * 100
    ).round(1)

# Modify original nation comparison table with lower and upper CI values
q5_nations_pct_df_ci = q5_nations_pct_df.copy(deep=True)

q5_nations_pct_df_ci = q5_nations_pct_df_ci.astype(str)

for index in q5_nations_pct_df_ci.index:
    for response in q5_nations_pct_df_ci.columns:
        lower = weighted_props_ci_nations[index.lower()].loc[response, "Lower CI"]
        upper = weighted_props_ci_nations[index.lower()].loc[response, "Upper CI"]
        q5_nations_pct_df_ci.loc[index, response] = (
            f"""{q5_nations_pct_df_ci.loc[index, response]} ({lower} - {upper})"""
        )

# Add sample sizes
q5_nations_pct_df_ci = (
    q5_nations_pct_df_ci.join(q5_nations_df["Nation total"])
    .assign(n=lambda df: df["Nation total"].round(0).astype(int))
    .drop(columns="Nation total")
)

# %%
# Format display
q5_nations_pct_df_ci.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) responses to 'How likely, if at all, would you be to consider installing a heat pump in your home as your main heating source?'"
)

# %% [markdown]
# ## RQ 1. By tenure

# %% [markdown]
# The tenure group with the highest proportion of people who said they definitely/probably would consider is the "Neither" group (living with family/friends, or other) (49%), compared to 42% of homeowners and 44% of renters.
#
# The group with the highest proportion of people who said they definitely/probably wouldn't consider is the homeowner group (50%). This is notably higher than the percentages of renters and others (37% and 30%, respectively).
#
# Unsurprisingly, the "neither" tenure group had the highest percentage of people who said they didn't know (21%), compared to homeowners (8%) and renters (18%).
#
# The homeowners group is the only one where the proportion of those who wouldn't consider (50%) is greater than the proportion who said they would (42%).
#

# %%
# Create Q5 response breakdown by tenure table for each nation
q5_tenure_tables = {}
for group, base_data in q5_base_data_lookup.items():
    tables = {}
    counts_by_tenure, proportions_by_tenure = summary.generate_tenure_breakdown(
        base_data, "PEN_Q5", q5_response_aggregation_dict, proportions_axis=1
    )
    tables["counts"] = counts_by_tenure
    tables["proportions"] = proportions_by_tenure

    q5_tenure_tables[group] = tables

# %%
# Plot broad tenure categories comparison
q5_tenure_pct_df = (
    q5_tenure_tables["uk"]["proportions"]
    .reindex(index=["Own (net)", "Rent (net)", "Neither (net)", "Other"])
    .drop(columns=["Total", "Would consider (net)", "Wouldn't consider (net)"])
)

summary.create_stacked_horizontal_bar_chart(
    q5_tenure_pct_df, "Percentage", "Tenure", 10, (10, 5)
).show()

# %%
# Add 95% confidence intervals for proportions

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

q5_tenure_proportions_with_ci = summary.modify_proportion_crosstab_with_ci(
    base_data=q5_base_data,
    proportions_df=q5_tenure_tables["uk"]["proportions"],
    aggregated_responses_column_dict=q5_response_aggregation_dict,
    aggregated_responses_index_dict=aggregated_responses_tenure,
    index_survey_code="profile_house_tenure",
).drop(columns="Total")

# %%
# Display broad tenure categories for UK, proportions with 95% CI
q5_tenure_proportions_with_ci_summary = (
    q5_tenure_proportions_with_ci.join(q5_tenure_tables["uk"]["counts"]["Total"])
    .assign(n=lambda df: df["Total"].round(0).astype(int))
    .drop(columns="Total")
)

# %%
# Format display
q5_tenure_proportions_with_ci_summary.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) responses to 'How likely, if at all, would you be to consider installing a heat pump in your home as your main heating source?'"
)

# %%
# Chi-squared contingency test across tenure
q5_tenure_counts_summary = q5_tenure_tables["uk"]["counts"].drop(
    index=["Own (net)", "Rent (net)", "Neither (net)", "Total"],
    columns=["Total", "Would consider (net)", "Wouldn't consider (net)"],
)
chi2_stat, p_val, dof, expected = chi2_contingency(q5_tenure_counts_summary)

if p_val < 0.05:
    print(
        f"A chi-squared test of independence found a significant association between tenure subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
else:
    print(
        f"A chi-squared test of independence showed no significant relationship between tenure subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )

# %% [markdown]
# ## RQ 1. By tenure: Homeowners

# %% [markdown]
# People who own their home outright have the lowest proportion of people who would consider installing a heat pump (36%) and the greatest proportion who say they wouldn't (58%), compared to people who own their home with a mortgage (50% would, 42% wouldn't) or through part-ownership (46% would, 49% wouldn't).
#
# N.B. 49% of those who own their home outright are aged 55+ and said they definitely/probably wouldn't consider installing a heat pump.

# %%
# Plot homeowners only comparison table
q5_tenure_homeowners_pct_df = (
    q5_tenure_tables["uk"]["proportions"]
    .reindex(
        index=[
            "Own - outright",
            "Own - with a mortgage",
            "Own (part-own) - through shared ownership scheme (i.e. pay part mortgage, part rent)",
        ]
    )
    .drop(columns=["Total", "Would consider (net)", "Wouldn't consider (net)"])
    .rename(
        index={
            "Own (part-own) - through shared ownership scheme (i.e. pay part mortgage, part rent)": "Own (part-own)"
        }
    )
)

summary.create_stacked_horizontal_bar_chart(
    q5_tenure_homeowners_pct_df, "Percentage", "Homeowner tenure subgroup", 10, (10, 5)
).show()

# %%
# Display owner tenure categories for UK, proportions with 95% CI
q5_homeowner_proportions_with_ci_summary = (
    q5_tenure_proportions_with_ci.reindex(
        [
            "Own - outright",
            "Own - with a mortgage",
            "Own (part-own) - through shared ownership scheme (i.e. pay part mortgage, part rent)",
        ]
    )
    .join(q5_tenure_tables["uk"]["counts"]["Total"])
    .assign(n=lambda df: df["Total"].round(0).astype(int))
    .drop(columns="Total")
)

# %%
q5_homeowner_proportions_with_ci_summary.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) responses to 'How likely, if at all, would you be to consider installing a heat pump in your home as your main heating source?'"
)

# %%
# Chi-squared contingency test across owners tenures
q5_homeowner_counts_summary = (
    q5_tenure_tables["uk"]["counts"]
    .reindex(
        index=[
            "Own - outright",
            "Own - with a mortgage",
            "Own (part-own) - through shared ownership scheme (i.e. pay part mortgage, part rent)",
        ]
    )
    .drop(columns=["Total", "Would consider (net)", "Wouldn't consider (net)"])
)
chi2_stat, p_val, dof, expected = chi2_contingency(q5_homeowner_counts_summary)

if p_val < 0.05:
    print(
        f"A chi-squared test of independence found a significant association between tenure subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
else:
    print(
        f"A chi-squared test of independence showed no significant relationship between tenure subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )

# %%
# Run this cell to check age x tenure assertion
# (
#     pd.crosstab(
#         index=[q5_base_data["PEN_Q5"], q5_base_data["profile_julesage"]],
#         columns=q5_base_data["profile_house_tenure"],
#         values=q5_base_data["weight"],
#         aggfunc="sum",
#         margins=True,
#         normalize="columns",
#     )
#     * 100
# ).round(1)

# %% [markdown]
# ## RQ 1. By tenure: Renters

# %% [markdown]
# The renter tenure group with the greatest proportion who said they would consider installing a heat pump is private renters (50%), compared to the groups of people renting social housing (36% for local authority rentals and 38% for housing association rentals).
#
# Private renters also have the smallest proportion of those who said they wouldn't consider (34%), around ~10 percentage points lower than the groups of people renting social housing (43% for local authority rentals and 41% for housing association rentals).

# %%
# Plot homeowners only comparison table
q5_tenure_renters_pct_df = (
    q5_tenure_tables["uk"]["proportions"]
    .reindex(
        index=[
            "Rent - from a private landlord",
            "Rent - from my local authority",
            "Rent - from a housing association",
        ]
    )
    .drop(columns=["Total", "Would consider (net)", "Wouldn't consider (net)"])
)

summary.create_stacked_horizontal_bar_chart(
    q5_tenure_renters_pct_df, "Percentage", "Renter tenure subgroup", 10, (10, 5)
).show()

# %%
# Display renter tenure categories for UK, proportions with 95% CI
q5_renter_proportions_with_ci_summary = (
    q5_tenure_proportions_with_ci.reindex(
        [
            "Rent - from a private landlord",
            "Rent - from my local authority",
            "Rent - from a housing association",
        ]
    )
    .join(q5_tenure_tables["uk"]["counts"]["Total"])
    .assign(n=lambda df: df["Total"].round(0).astype(int))
    .drop(columns="Total")
)

# %%
q5_renter_proportions_with_ci_summary.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) responses to 'How likely, if at all, would you be to consider installing a heat pump in your home as your main heating source?'"
)

# %%
# Chi-squared contingency test across renter tenures
q5_renter_counts_summary = (
    q5_tenure_tables["uk"]["counts"]
    .reindex(
        index=[
            "Rent - from a private landlord",
            "Rent - from my local authority",
            "Rent - from a housing association",
        ]
    )
    .drop(columns=["Total", "Would consider (net)", "Wouldn't consider (net)"])
)
chi2_stat, p_val, dof, expected = chi2_contingency(q5_renter_counts_summary)

if p_val < 0.05:
    print(
        f"A chi-squared test of independence found a significant association between tenure subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
else:
    print(
        f"A chi-squared test of independence showed no significant relationship between tenure subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )

# %% [markdown]
# ## RQ 1. By tenure: Other

# %% [markdown]
# Approximately half of respondents who live with family/friends said they would consider installing a heat pump in their home (50%, 48% of those living with family/friends paying rent and those living rent-free, respectively).

# %%
# Plot neither/others comparison table
q5_tenure_other_pct_df = (
    q5_tenure_tables["uk"]["proportions"]
    .reindex(
        index=[
            "Neither - I live with my parents, family or friends but pay some rent to them",
            "Neither - I live rent-free with my parents, family or friends",
            "Other",
        ]
    )
    .drop(columns=["Total", "Would consider (net)", "Wouldn't consider (net)"])
)

summary.create_stacked_horizontal_bar_chart(
    q5_tenure_other_pct_df, "Percentage", "Other tenure subgroup", 10, (10, 5)
).show()

# %%
# Display neither/other tenure categories for UK, proportions with 95% CI
q5_other_proportions_with_ci_summary = (
    q5_tenure_proportions_with_ci.reindex(
        [
            "Neither - I live with my parents, family or friends but pay some rent to them",
            "Neither - I live rent-free with my parents, family or friends",
            "Other",
        ]
    )
    .join(q5_tenure_tables["uk"]["counts"]["Total"])
    .assign(n=lambda df: df["Total"].round(0).astype(int))
    .drop(columns="Total")
)

# %%
q5_other_proportions_with_ci_summary.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) responses to 'How likely, if at all, would you be to consider installing a heat pump in your home as your main heating source?'"
)

# %%
# Chi-squared contingency test across other/neither tenures
q5_other_counts_summary = (
    q5_tenure_tables["uk"]["counts"]
    .reindex(
        index=[
            "Neither - I live with my parents, family or friends but pay some rent to them",
            "Neither - I live rent-free with my parents, family or friends",
            # "Other",
        ]
    )
    .drop(columns=["Total", "Would consider (net)", "Wouldn't consider (net)"])
)
chi2_stat, p_val, dof, expected = chi2_contingency(q5_other_counts_summary)

if p_val < 0.05:
    print(
        f"A chi-squared test of independence found a significant association between tenure subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
else:
    print(
        f"A chi-squared test of independence showed no significant relationship between tenure subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )

# %% [markdown]
# ## RQ 1. By household income (1)

# %% [markdown]
# The proportion of respondents who said "Don't know" decreases with increasing income brackets - i.e., the higher the household income, the more likely the respondent is to have an opinion or know whether they or wouldn't consider installing a heat pump.
#
# There is an association between income and the proportion who said they would consider. Greater shares of respondents of higher income brackets who said they would consider compared to lower brackets - e.g. 39% of those earning between £10,000 and £29,999 per year vs. 59% of those earning between £100,000 and £149,999 per year.
#
# There isn't, however, as clear a trend between income and the proportion who said they wouldn't consider.
#

# %%
# Create Q5 response breakdown by tenure table for each nation
q5_income_tables = {}
for group, base_data in q5_base_data_lookup.items():
    tables = {}
    counts_by_income, proportions_by_income = summary.generate_income_breakdown(
        base_data,
        "PEN_Q5",
        q5_response_aggregation_dict,
        proportions_axis=1,
        aggregate_income=True,
    )
    tables["counts"] = counts_by_income
    tables["proportions"] = proportions_by_income

    q5_income_tables[group] = tables

# %%
# Plot income bracket comparison table
q5_income_pct_df = (
    q5_income_tables["uk"]["proportions"]
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
    .drop(columns=["Total", "Would consider (net)", "Wouldn't consider (net)"])
)

summary.create_stacked_horizontal_bar_chart(
    q5_income_pct_df, "Percentage", "Income", 10, (10, 10)
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

q5_income_proportions_with_ci = summary.modify_proportion_crosstab_with_ci(
    base_data=q5_base_data,
    proportions_df=q5_income_tables["uk"]["proportions"],
    aggregated_responses_column_dict=q5_response_aggregation_dict,
    aggregated_responses_index_dict=aggregated_responses_income,
    index_survey_code="profile_gross_household",
).drop(columns="Total")

# %%
# Show only aggregated income brackets
q5_income_proportions_with_ci_summary = (
    q5_income_proportions_with_ci.reindex(
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
    .join(q5_income_tables["uk"]["counts"]["Total"].round(0))
    .rename(columns={"Total": "n"})
    .assign(n=lambda df: df["n"].astype(int))
)

# %%
q5_income_proportions_with_ci_summary.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) responses to 'How likely, if at all, would you be to consider installing a heat pump in your home as your main heating source?'"
)

# %%
# Chi-squared contingency test across income brackets (net)
q5_income_counts_summary = (
    q5_income_tables["uk"]["counts"]
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
    .drop(columns=["Total", "Would consider (net)", "Wouldn't consider (net)"])
)
chi2_stat, p_val, dof, expected = chi2_contingency(q5_income_counts_summary)

if p_val < 0.05:
    print(
        f"A chi-squared test of independence found a significant association between income group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
else:
    print(
        f"A chi-squared test of independence showed no significant relationship between income group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )

# %% [markdown]
# ## RQ 1. By household income (2)

# %%
df = q5_income_tables["uk"]["proportions"].reindex(
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
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

highlight_category = ["Don't know", "Prefer not to answer"]
bar_colours = [
    "grey" if category in highlight_category else "darkgreen" for category in df.index
]
ax = sns.barplot(
    x="profile_gross_household", y="Would consider (net)", data=df, palette=bar_colours
)

for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.0f}",  # Add label with one decimal place
        (
            p.get_x() + p.get_width() / 2.0,
            p.get_height(),
        ),
        ha="center",
        va="center",
        fontsize=9,
        xytext=(0, 5),
        textcoords="offset points",
    )

plt.xlabel("Gross household income", fontsize=10)
plt.ylabel(
    "Percentage of income group that would\nconsider installing a heat pump (%)",
    fontsize=9,
)

plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(0, 80)
plt.title("Income group vs. Proportion who would consider installing a heat pump")
plt.tight_layout()
plt.show()

# %%
df = q5_income_tables["uk"]["proportions"].reindex(
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
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

highlight_category = ["Don't know", "Prefer not to answer"]
bar_colours = [
    "grey" if category in highlight_category else "darkblue" for category in df.index
]
ax = sns.barplot(
    x="profile_gross_household",
    y="Wouldn't consider (net)",
    data=df,
    palette=bar_colours,
)

for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.0f}",  # Add label with one decimal place
        (
            p.get_x() + p.get_width() / 2.0,
            p.get_height(),
        ),
        ha="center",
        va="center",
        fontsize=9,
        xytext=(0, 5),
        textcoords="offset points",
    )

plt.xlabel("Gross household income", fontsize=10)
plt.ylabel(
    "Percentage of income group that wouldn't\nconsider installing a heat pump (%)",
    fontsize=9,
)

plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(0, 80)
plt.title("Income group vs. Proportion who wouldn't consider installing a heat pump")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## RQ 1. By age (1)

# %% [markdown]
# The age group with the highest proportion who would consider installing a heat pump (55%) is 25-34. Conversely, the age group with the highest proportion who wouldn't consider installing a heat pump (57%) is 55+.

# %%
# Create Q5 response breakdown by age table for the UK
q5_age_tables = {}

# Counts
q5_age_tables["counts"] = pd.crosstab(
    index=q5_base_data["profile_julesage"],
    columns=q5_base_data["PEN_Q5"],
    values=q5_base_data["weight"],
    aggfunc="sum",
    margins=True,
    margins_name="Total",
)

# Proportions
q5_age_tables["proportions"] = pd.crosstab(
    index=q5_base_data["profile_julesage"],
    columns=q5_base_data["PEN_Q5"],
    values=q5_base_data["weight"],
    aggfunc="sum",
    normalize="index",
)
q5_age_tables["proportions"]["Total"] = q5_age_tables["proportions"].sum(axis=1)
q5_age_tables["proportions"] = (q5_age_tables["proportions"] * 100).round(1)

# Add aggregated responses
for df in q5_age_tables.values():
    for combined_col, source_cols in q5_response_aggregation_dict.items():
        df[combined_col] = df[source_cols].sum(axis=1)
    df.drop(columns="Not asked", inplace=True)

# %%
age_df = q5_age_tables["proportions"].drop(
    columns=["Would consider (net)", "Wouldn't consider (net)", "Total"]
)
summary.create_stacked_horizontal_bar_chart(
    age_df, "Percentage", "Age", 10, (10, 5)
).show()

# %%
# 95% confidence intervals for proportions
q5_age_proportions_with_ci = summary.modify_proportion_crosstab_with_ci(
    base_data=q5_base_data,
    proportions_df=q5_age_tables["proportions"],
    aggregated_responses_column_dict=q5_response_aggregation_dict,
    aggregated_responses_index_dict={},
    index_survey_code="profile_julesage",
).drop(columns="Total")

# %%
q5_age_proportions_with_ci.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) responses to 'How likely, if at all, would you be to consider installing a heat pump in your home as your main heating source?'"
)

# %%
# Chi-squared contingency test across age groups
q5_age_counts_summary = q5_age_tables["counts"].drop(
    columns=["Total", "Would consider (net)", "Wouldn't consider (net)"],
    index="Total",
)
chi2_stat, p_val, dof, expected = chi2_contingency(q5_age_counts_summary)

if p_val < 0.05:
    print(
        f"A chi-squared test of independence found a significant association between age group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
else:
    print(
        f"A chi-squared test of independence showed no significant relationship between age group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )

# %% [markdown]
# ## RQ 1. By age (2)

# %% [markdown]
# There is an association between age and likelihood of considering/not considering a heat pump, with trend being starker when looking at the percentage of respondents who said they wouldn't consider of each age group (28% of 18-24 year olds to 57% of 55+ year olds).

# %%
sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))

ax = sns.barplot(
    x="profile_julesage",
    y="Would consider (net)",
    data=q5_age_tables["proportions"],
    color="darkgreen",
)

for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.0f}",  # Add label with one decimal place
        (
            p.get_x() + p.get_width() / 2.0,
            p.get_height(),
        ),
        ha="center",
        va="center",
        fontsize=9,
        xytext=(0, 5),
        textcoords="offset points",
    )

plt.xlabel("Age group", fontsize=10)
plt.ylabel(
    "Percentage of age group that would\nconsider installing a heat pump (%)",
    fontsize=9,
)

plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(0, 80)
plt.title("Age group vs. Proportion who would consider installing a heat pump")
plt.tight_layout()
plt.show()

# %%
sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))

ax = sns.barplot(
    x="profile_julesage",
    y="Wouldn't consider (net)",
    data=q5_age_tables["proportions"],
    color="darkblue",
)

for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.0f}",  # Add label with one decimal place
        (
            p.get_x() + p.get_width() / 2.0,
            p.get_height(),
        ),
        ha="center",
        va="center",
        fontsize=9,
        xytext=(0, 5),
        textcoords="offset points",
    )

plt.xlabel("Age group", fontsize=10)
plt.ylabel(
    "Percentage of age group that wouldn't\nconsider installing a heat pump (%)",
    fontsize=9,
)

plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(0, 80)
plt.title("Age group vs. Proportion who wouldn't consider installing a heat pump")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## RQ 1. Whether home has energy tech, space/infrastructure or boiler replacement opportunity

# %% [markdown]
# Most people with a type of smart energy technology (i.e. home battery, electric vehicle, electric vehicle charger, solar panels) said they would consider installing a heat pump (55-63%). But only a small percentage of the population have these technologies (2-8%).
#
# The largest group by far with a particular thing in their home already is the group with private outdoor space (56% of the UK population), where equal proportions of people said they would (46%) and wouldn't consider (46%) a heat pump.

# %%
# Compile all options
q3_options = [
    option
    for option in code_question_lookup.keys()
    if "Q3_" in option and "collapsed" not in option
]

# Generate counts, proportions and proportions with 95% CI tables
q3_option_tables = {}
q3_option_tables_with_ci = {}
for option in q3_options:
    # Cast different data type for compatibility with processing
    q5_base_data.loc[:, option] = q5_base_data[option].astype(str)
    q5_base_data.loc[:, option] = pd.Categorical(
        q5_base_data[option], categories=["True", "False"], dtype="category"
    )
    q3_option_tables[option], q3_option_tables_with_ci[option] = (
        summary.generate_q5_outputs(q5_base_data, option, drop_not_asked=False)
    )

# %%
# Concatenate all counts, proportions, and proportions_with_ci, dropping unwanted indices
q3_counts = pd.concat(
    [
        q3_option_tables[opt]["counts"].drop(index=["False", "Total"], errors="ignore")
        for opt in q3_options
    ],
    axis=0,
)

q3_proportions = pd.concat(
    [
        q3_option_tables[opt]["proportions"].drop(index=["False"], errors="ignore")
        for opt in q3_options
    ],
    axis=0,
)

q3_proportions_with_ci = pd.concat(
    [
        q3_option_tables_with_ci[opt].drop(index=["False"], errors="ignore")
        for opt in q3_options
    ],
    axis=0,
)

# Create option names from lookup
option_names = [
    (
        code_question_lookup.get(opt, "").split(":", 1)[1].strip()
        if ":" in code_question_lookup.get(opt, "")
        else code_question_lookup.get(opt, "")
    )
    for opt in q3_options
]

# Assign indices
q3_counts.index = option_names
q3_proportions.index = option_names
q3_proportions_with_ci.index = option_names

# %%
# Display reordered summary table
q3_proportions_summary = q3_proportions_with_ci.sort_values(
    by="Would consider (net)", ascending=False
)[["Would consider (net)", "Wouldn't consider (net)", "Don't know"]]
q3_proportions_summary = q3_proportions_summary.join(q3_counts["Total"].astype(int))
q3_proportions_summary = q3_proportions_summary.rename(
    columns={"Total": "n (% of population)"}
)
q3_proportions_summary["n (% of population)"] = q3_proportions_summary[
    "n (% of population)"
].apply(lambda x: f"{x} ({(x/6875)*100:.1f}%)")

# %%
q3_proportions_summary.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) responses to 'How likely, if at all, would you be to consider installing a heat pump in your home as your main heating source?'"
)

# %%
# Chi-squared contingency test across groups
chi2_stat, p_val, dof, expected = chi2_contingency(
    q3_counts.drop(columns=["Total", "Would consider (net)", "Wouldn't consider (net)"])
)

if p_val < 0.05:
    print(
        f"A chi-squared test of independence found a significant association between group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
else:
    print(
        f"A chi-squared test of independence showed no significant relationship between group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )

# %% [markdown]
# ## RQ 1. By awareness

# %% [markdown]
# People who have heard of heat pumps and know what they are have the greatest proportion of those who wouldn't consider installing a heat pump (51%; 31% who said definitely wouldn't consider and 20% who said probably wouldn't consider).
#
# The proportions who would consider a heat pump are similar for groups who know what heat pumps are (44%) and who don't know what heat pumps are (46%).
#
# Interestingly, knowing/not knowing what heat pumps are doesn't seem to be associated with whether or not someone would consider them as their main heating source - i.e. 44% of people who know what they are vs. 46% of those who don't know what they are.
#
# Strangely, the group with the greatest proportion of those who wouldn't consider is the group who does know what they are (51%). This proportion is a notably higher than the groups who have heard of them but don't know what they are (38%) and have never heard of them (31%).
# - Within the group who know what they are, the most common response was "*Definitely" wouldn't consider" (31%).
#
# For those who have never heard of heat pumps, there's a roughly even split between those who would (35%) and wouldn't consider (31%).

# %%
# Create counts and proportions tables
q5_q4_tables, q5_q4_proportions_with_ci = summary.generate_q5_outputs(
    q5_base_data, "PEN_Q4"
)

# %%
q4_df = q5_q4_tables["proportions"].drop(
    columns=["Would consider (net)", "Wouldn't consider (net)", "Total"]
)

summary.create_stacked_horizontal_bar_chart(
    q4_df,
    "Percentage",
    "Before taking this survey, had you ever\nheard of heat pumps as a home heating system?",
    10,
    (10, 5),
).show()

# %%
q5_q4_proportions_with_ci.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) responses to 'How likely, if at all, would you be to consider installing a heat pump in your home as your main heating source?'"
)

# %%
# Chi-squared contingency test across Q4 response groups
q5_q4_counts_summary = q5_q4_tables["counts"].drop(
    columns=["Total", "Would consider (net)", "Wouldn't consider (net)"],
    index=["Total", "Not asked"],
)
chi2_stat, p_val, dof, expected = chi2_contingency(q5_q4_counts_summary)

if p_val < 0.05:
    print(
        f"A chi-squared test of independence found a significant association between group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
else:
    print(
        f"A chi-squared test of independence showed no significant relationship between group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )

# %% [markdown]
# ## RQ 1. By whether they had seen a heat pump

# %% [markdown]
# The group with the highest proportion of people who said they would consider a heat pump was the group who hadn't seen one before (47%), compared to those who had (42%).
#
# Respondents who had seen a heat pump before have the highest proportion of people who said they wouldn't consider a heat pump (53%), compared to 39% of those who hadn't.
#
# *It seems like the respondents who know the most about them (know how they work and/or have seen one before) are the ones who are more likely to know they definitely wouldn't consider installing a heat pump in their home.*

# %%
# Create counts and proportions tables
# 739 not asked (weighted) as they have a heat pump or said they had never heard of heat pumps
q5_q6_tables, q5_q6_proportions_with_ci = summary.generate_q5_outputs(
    q5_base_data, "PEN_Q6", drop_not_asked=False
)

# %%
q6_df = q5_q6_tables["proportions"].drop(
    columns=["Would consider (net)", "Wouldn't consider (net)", "Total"],
    index="Not asked",
)

summary.create_stacked_horizontal_bar_chart(
    q6_df,
    "Percentage",
    "Before taking this survey,\nhave you ever seen what a heat pump looks like?",
    10,
    (10, 5),
).show()

# %%
q5_q6_proportions_with_ci.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) responses to 'How likely, if at all, would you be to consider installing a heat pump in your home as your main heating source?'"
)

# %%
# Chi-squared contingency test across Q6 response groups
q5_q6_counts_summary = q5_q6_tables["counts"].drop(
    columns=["Total", "Would consider (net)", "Wouldn't consider (net)"],
    index=["Total", "Not asked"],
)
chi2_stat, p_val, dof, expected = chi2_contingency(q5_q6_counts_summary)

if p_val < 0.05:
    print(
        f"A chi-squared test of independence found a significant association between group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
else:
    print(
        f"A chi-squared test of independence showed no significant relationship between group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )

# %% [markdown]
# ## RQ 1. By whether they had been to a home with a heat pump

# %% [markdown]
# The proportion who would consider a heat pump of those who have been to a home with a heat pump (51%) is greater than the proportion of those who haven't (41%), but similar to the proportion of those who don't know/can't recall (54%).
#
# Similarly, the proportion who said they wouldn't consider a heat pump is lower for the group who have been to a home with a heat pump (45%) than in the group who haven't (50%).

# %%
# Create counts and proportions tables
# 739 not asked (weighted) as they have a heat pump or said they had never heard of heat pumps
q5_q7_tables, q5_q7_proportions_with_ci = summary.generate_q5_outputs(
    q5_base_data, "PEN_Q7", drop_not_asked=False
)

# %%
q7_df = q5_q7_tables["proportions"].drop(
    columns=["Would consider (net)", "Wouldn't consider (net)", "Total"],
    index="Not asked",
)

summary.create_stacked_horizontal_bar_chart(
    q7_df,
    "Percentage",
    "Have you ever been to a home that has a heat pump?",
    10,
    (10, 5),
).show()

# %%
q5_q7_proportions_with_ci.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) responses to 'How likely, if at all, would you be to consider installing a heat pump in your home as your main heating source?'"
)

# %%
# Chi-squared contingency test across Q7 response groups
q5_q7_counts_summary = q5_q7_tables["counts"].drop(
    columns=["Total", "Would consider (net)", "Wouldn't consider (net)"],
    index=["Total", "Not asked"],
)
chi2_stat, p_val, dof, expected = chi2_contingency(q5_q7_counts_summary)

if p_val < 0.05:
    print(
        f"A chi-squared test of independence found a significant association between group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
else:
    print(
        f"A chi-squared test of independence showed no significant relationship between group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )

# %% [markdown]
# ## RQ 1. By whether family/friends/neighbours own a heat pump

# %% [markdown]
# Most people who have family/friends/neighbours with a heat pump would consider installing one in their own home (58%). This proportion is notably higher than in those who don't (41%).

# %%
# Create counts and proportions tables
# 739 not asked (weighted) as they have a heat pump or said they had never heard of heat pumps
q5_q8_tables, q5_q8_proportions_with_ci = summary.generate_q5_outputs(
    q5_base_data, "PEN_Q8", drop_not_asked=False
)

# %%
q8_df = q5_q8_tables["proportions"].drop(
    columns=["Would consider (net)", "Wouldn't consider (net)", "Total"],
    index="Not asked",
)

summary.create_stacked_horizontal_bar_chart(
    q8_df,
    "Percentage",
    "Does anyone in your family, close friends\nor neighbours own a heat pump?",
    10,
    (10, 5),
).show()

# %%
q5_q8_proportions_with_ci.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) responses to 'How likely, if at all, would you be to consider installing a heat pump in your home as your main heating source?'"
)

# %%
# Chi-squared contingency test across Q8 response groups
q5_q8_counts_summary = q5_q8_tables["counts"].drop(
    columns=["Total", "Would consider (net)", "Wouldn't consider (net)"],
    index=["Total", "Not asked"],
)
chi2_stat, p_val, dof, expected = chi2_contingency(q5_q8_counts_summary)

if p_val < 0.05:
    print(
        f"A chi-squared test of independence found a significant association between group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
else:
    print(
        f"A chi-squared test of independence showed no significant relationship between group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )

# %% [markdown]
# ## RQ 1. By whether they have had a conversation with family/friends/neighbours about installing a heat pump

# %% [markdown]
# Most people who have had a conversation about installing a heat pump would consider one (57%) This proportion is greater than the proportions of those who haven't (41%).
#
# Similarly, a lower proportion of those who have had a conversation said they wouldn't consider (40%), compared to those who haven't had a conversation (48%).

# %%
# Create counts and proportions tables
# 739 not asked (weighted) as they have a heat pump or said they had never heard of heat pumps
q5_q9_tables, q5_q9_proportions_with_ci = summary.generate_q5_outputs(
    q5_base_data, "PEN_Q9", drop_not_asked=False
)

# %%
q9_df = q5_q9_tables["proportions"].drop(
    columns=["Would consider (net)", "Wouldn't consider (net)", "Total"],
    index="Not asked",
)
summary.create_stacked_horizontal_bar_chart(
    q9_df,
    "Percentage",
    "Have you ever had a conversation with your family, close friends\nor neighbours about installing a heat pump in your home?",
    10,
    (10, 5),
).show()

# %%
q5_q9_proportions_with_ci.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) responses to 'How likely, if at all, would you be to consider installing a heat pump in your home as your main heating source?'"
)

# %%
# Chi-squared contingency test across Q9 response groups
q5_q9_counts_summary = q5_q9_tables["counts"].drop(
    columns=["Total", "Would consider (net)", "Wouldn't consider (net)"],
    index=["Total", "Not asked"],
)
chi2_stat, p_val, dof, expected = chi2_contingency(q5_q9_counts_summary)

if p_val < 0.05:
    print(
        f"A chi-squared test of independence found a significant association between group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
else:
    print(
        f"A chi-squared test of independence showed no significant relationship between group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )

# %% [markdown]
# ## Summarising the effects of prior knowledge/experiences

# %% [markdown]
# **Know what they are**
# - Know what they are: 44% would consider vs. 51% wouldn't
# - Have heard of but don't know what they are: 46% vs. 38%
# - Never heard of: 35% vs. 31%
#
# *Knowing what they are does not lead to a greater likelihood of considering.*
#
# **Have seen one**
# - Seen: 42% vs. 53%
# - Not seen: 47% vs. 39%
#
# *Having seen one does not lead to a greater likelihood of considering.*
#
# **Been to a home with one**
# - Been: 51% vs. 45%
# - Not been: 41% vs. 50%
#
# *Having been to a home with a heat pump does lead to a greater likelihood of considering.*
#
# **Family/friends/neighbours have one**
# - Have: 58% vs. 40%
# - Don't have: 41% vs. 51%
#
# *Having family/friends/neighbours with a heat pump does lead to a greater likelihood of considering.*
#
# **Have had a conversation about installing one**
# - Have: 57% vs. 40%
# - Haven't: 41% vs. 48%
#
# *Having a conversation with family/friends/neighbours about installing a heat pump does lead to a greater likelihood of considering.*
#

# %% [markdown]
# ## RQ 1. By affordability: Without additional borrowing

# %% [markdown]
# Unsurprisingly, the groups who said that a heat pump would be very or fairly affordable had greater proportions who said that they are likely to consider a heat pump (64% and 68%, respectively), compared to those who said it would be very or fairly unaffordable (57%, 34%).
#
# The split between groups seems to be between the "Fairly unaffordable" and "Very unaffordable" groups - i.e., percentage who would consider of the fairly unaffordable group is closer to the percentages in very/fairly affordable than to the very unaffordable percentage.
#

# %%
# Create counts and proportions tables
q5_q5a_tables, q5_q5a_proportions_with_ci = summary.generate_q5_outputs(
    q5_base_data, "PEN_Q5Anew", drop_not_asked=True
)

# %%
q5a_df = q5_q5a_tables["proportions"].drop(
    columns=["Would consider (net)", "Wouldn't consider (net)", "Total"],
)

summary.create_stacked_horizontal_bar_chart(
    q5a_df,
    "Percentage",
    "To what extent, if at all, would installing a heat pump be\naffordable for your household, without taking out additional borrowing?",
    10,
    (10, 5),
).show()

# %%
q5_q5a_proportions_with_ci = (
    q5_q5a_proportions_with_ci.join(q5_q5a_tables["counts"]["Total"])
    .rename(columns={"Total": "n"})
    .assign(n=lambda df: df["n"].astype(int))
)

q5_q5a_proportions_with_ci.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) responses to 'How likely, if at all, would you be to consider installing a heat pump in your home as your main heating source?'"
)

# %%
# Chi-squared contingency test across Q5A response groups
q5_q5a_counts_summary = q5_q5a_tables["counts"].drop(
    columns=["Total", "Would consider (net)", "Wouldn't consider (net)"],
    index=["Total", "Not asked"],
)
chi2_stat, p_val, dof, expected = chi2_contingency(q5_q5a_counts_summary)

if p_val < 0.05:
    print(
        f"A chi-squared test of independence found a significant association between group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
else:
    print(
        f"A chi-squared test of independence showed no significant relationship between group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )

# %% [markdown]
# ## RQ 1. By willingness to borrow: Mortgage provider

# %% [markdown]
# Respondents who are more willing to borrow funds via a mortgage provider had higher proportions who said they would consider installing a heat pump. For instance, almost all who were very willing to borrow through a mortgage provider said they would consider (98%).

# %%
# Create counts and proportions tables
# Homeowners only
q5_q5b_tables, q5_q5b_proportions_with_ci = summary.generate_q5_outputs(
    q5_base_data, "PEN_Q5Bnew", drop_not_asked=False
)

# %%
q5b_df = q5_q5b_tables["proportions"].drop(
    columns=[
        "Would consider (net)",
        "Wouldn't consider (net)",
        "Total",
    ],
    index="Not asked",
)

summary.create_stacked_horizontal_bar_chart(
    q5b_df,
    "Percentage",
    "How willing, if at all, would you be to borrow funds from\na mortgage providerto fund the installation of\nlow carbon heating (such as heat pumps) in your home?",
    8,
    (10, 5),
).show()

# %%
q5_q5b_proportions_with_ci = (
    q5_q5b_proportions_with_ci.join(q5_q5b_tables["counts"]["Total"])
    .rename(columns={"Total": "n"})
    .assign(n=lambda df: df["n"].astype(int))
)

q5_q5b_proportions_with_ci.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) responses to 'How likely, if at all, would you be to consider installing a heat pump in your home as your main heating source?'"
)

# %%
# Chi-squared contingency test across Q5A response groups
q5_q5b_counts_summary = q5_q5b_tables["counts"].drop(
    columns=["Total", "Would consider (net)", "Wouldn't consider (net)"],
    index=["Total", "Not asked"],
)
chi2_stat, p_val, dof, expected = chi2_contingency(q5_q5b_counts_summary)

if p_val < 0.05:
    print(
        f"A chi-squared test of independence found a significant association between group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
else:
    print(
        f"A chi-squared test of independence showed no significant relationship between group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )

# %% [markdown]
# ## RQ 1. By willingness to borrow: Unsecured borrowing

# %% [markdown]
# Similarly, respondents who are more willing to borrow funds through unsecured borrowing had higher proportions who said they would consider installing a heat pump.

# %%
# Create counts and proportions tables
# Homeowners only
q5_q5c_tables, q5_q5c_proportions_with_ci = summary.generate_q5_outputs(
    q5_base_data, "PEN_Q5Cnew", drop_not_asked=False
)

# %%
q5c_df = q5_q5c_tables["proportions"].drop(
    columns=[
        "Would consider (net)",
        "Wouldn't consider (net)",
        "Total",
    ],
    index="Not asked",
)

summary.create_stacked_horizontal_bar_chart(
    q5c_df,
    "Percentage",
    "How willing, if at all, would you be to take out unsecured borrowing\n(e.g. via a personal loan or credit card) in order to fund the installation of\nlow carbon heating (such as heat pumps) in your home?",
    8,
    (10, 5),
).show()

# %%
q5_q5c_proportions_with_ci = (
    q5_q5c_proportions_with_ci.join(q5_q5c_tables["counts"]["Total"])
    .rename(columns={"Total": "n"})
    .assign(n=lambda df: df["n"].astype(int))
)

q5_q5c_proportions_with_ci.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) responses to 'How likely, if at all, would you be to consider installing a heat pump in your home as your main heating source?'"
)

# %%
# Chi-squared contingency test across Q5A response groups
q5_q5c_counts_summary = q5_q5c_tables["counts"].drop(
    columns=["Total", "Would consider (net)", "Wouldn't consider (net)"],
    index=["Total", "Not asked"],
)
chi2_stat, p_val, dof, expected = chi2_contingency(q5_q5c_counts_summary)

if p_val < 0.05:
    print(
        f"A chi-squared test of independence found a significant association between group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
else:
    print(
        f"A chi-squared test of independence showed no significant relationship between group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )

# %%
## RQ 1. By desire to buy a house with a heat pump

# %%
# Create counts and proportions tables
# q5_q5d_tables, q5_q5d_proportions_with_ci = summary.generate_q5_outputs(
#     q5_base_data, "PEN_Q5Dnew", drop_not_asked=False
# )

# %%
# fontsize = 8
# q5d_df = q5_q5d_tables["proportions"].drop(
#     columns=[
#         "Would consider (net)",
#         "Wouldn't consider (net)",
#         "Total",
#     ],
# )
# ax = q5d_df.plot(kind="barh", stacked=True, figsize=(10, 5), colormap="Dark2")
# ax.set_xlabel("Percentage", fontsize=fontsize)
# ax.set_ylabel(
#     "How desirable, if at all, would buying a house that has a heat pump be for you?",
#     fontsize=fontsize,
# )
# ax.tick_params(axis="x", labelsize=fontsize)
# ax.tick_params(axis="y", labelsize=fontsize)
# ax.legend(loc="center", bbox_to_anchor=(0.5, 1.1), ncol=3, fontsize=fontsize)
# ax.set_xlim(0, 100)
# ax.invert_yaxis()

# # Add labels to each segment
# for c in ax.containers:
#     for rect in c:
#         width = rect.get_width()
#         x_position = rect.get_x() + width / 2
#         y_position = rect.get_y() + rect.get_height() / 2
#         ax.text(
#             x_position,
#             y_position,
#             f"{width:.1f}%",
#             ha="center",
#             va="center",
#             fontsize=fontsize,
#             color="white",
#         )
# plt.tight_layout()
# plt.show()

# %%
# q5_q5d_proportions_with_ci = q5_q5d_proportions_with_ci.join(
#     q5_q5d_tables["counts"]["Total"]
# )
# q5_q5d_proportions_with_ci = q5_q5d_proportions_with_ci.rename(columns={"Total": "n"})
# q5_q5d_proportions_with_ci["n"] = q5_q5d_proportions_with_ci["n"].astype(int)
# q5_q5d_proportions_with_ci

# %%
# # Chi-squared contingency test across Q5A response groups
# q5_q5d_counts_summary = q5_q5d_tables["counts"].drop(
#     columns=["Total", "Would consider (net)", "Wouldn't consider (net)"],
#     index=["Total"],
# )
# chi2_stat, p_val, dof, expected = chi2_contingency(q5_q5d_counts_summary)

# if p_val < 0.05:
#     print(
#         f"A chi-squared test of independence found a significant association between group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
#     )
# else:
#     print(
#         f"A chi-squared test of independence showed no significant relationship between group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
#     )

# %% [markdown]
# ## RQ 1. By main perceived barrier

# %% [markdown]
# The question that respondents were asked was: Which ONE of the following factors do you think would prevent you the MOST from installing a heat pump in your home?

# %% [markdown]
# *This breakdown of results is a little tricky to analyse because we can't be certain it is the reason why they said they wouldn't consider installing a heat pump.*
#
# Example statement:
# - Of the people who said that skepticism over heat pump technology would be the factor that would prevent them the most from installing a heat pump, 72% said they are not likely to consider installing heat pump.
#
# Note: Q12 (main barrier) was asked *after* Q5 (would you consider).

# %%
# Create counts and proportions tables
# Homeowners only
q5_q12_tables, q5_q12_proportions_with_ci = summary.generate_q5_outputs(
    q5_base_data, "PEN_Q12", drop_not_asked=False
)

# %%
q5_q12_proportions_with_ci = (
    q5_q12_proportions_with_ci.join(q5_q12_tables["counts"]["Total"])
    .drop(index="Not asked", errors="ignore")
    .rename(columns={"Total": "n"})
    .assign(n=lambda df: df["n"].astype(int))
    .sort_values(by="Wouldn't consider (net)", ascending=False)[
        ["Would consider (net)", "Wouldn't consider (net)", "n"]
    ]
)

q5_q12_proportions_with_ci.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) responses to 'How likely, if at all, would you be to consider installing a heat pump in your home as your main heating source?'"
)

# %%
# Chi-squared contingency test across Q5A response groups
q5_q12_counts_summary = q5_q12_tables["counts"].drop(
    columns=["Total", "Would consider (net)", "Wouldn't consider (net)"],
    index=["Total", "Not asked"],
)
chi2_stat, p_val, dof, expected = chi2_contingency(q5_q12_counts_summary)

if p_val < 0.05:
    print(
        f"A chi-squared test of independence found a significant association between group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
else:
    print(
        f"A chi-squared test of independence showed no significant relationship between group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )

# %% [markdown]
# ## RQ 1. By main perceived benefit

# %% [markdown]
# The question that respondents were asked: Which ONE of the following benefits about heat pumps is MOST likely to encourage you to install a heat pump in your home in the future?

# %% [markdown]
# *Similarly to the barrier question, this breakdown of results is a little tricky to analyse because we can't be certain it is the reason why they said they would consider installing a heat pump.*
#
# Example statement:
# - Of the people who said that keeping up with other people would be the factor that would encourage them the most from installing a heat pump, 82% said they are likely to consider installing heat pump.
#
# Note: Q14 (main benefit) was asked *after* Q5 (would you consider).

# %%
# Create counts and proportions tables
# Homeowners only
q5_q14_tables, q5_q14_proportions_with_ci = summary.generate_q5_outputs(
    q5_base_data, "PEN_Q14", drop_not_asked=False
)

# %%
q5_q14_proportions_with_ci = (
    q5_q14_proportions_with_ci.join(q5_q14_tables["counts"]["Total"])
    .drop(index="Not asked", errors="ignore")
    .rename(columns={"Total": "n"})
    .assign(n=lambda df: df["n"].astype(int))
    .sort_values(by="Would consider (net)", ascending=False)[
        ["Would consider (net)", "Wouldn't consider (net)", "n"]
    ]
)

q5_q14_proportions_with_ci.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "Proportions of subpopulation (%) (95% CI) responses to 'How likely, if at all, would you be to consider installing a heat pump in your home as your main heating source?'"
)

# %%
# Chi-squared contingency test across Q5A response groups
q5_q14_counts_summary = q5_q14_tables["counts"].drop(
    columns=["Total", "Would consider (net)", "Wouldn't consider (net)"],
    index=["Total", "Not asked"],
)
chi2_stat, p_val, dof, expected = chi2_contingency(q5_q14_counts_summary)

if p_val < 0.05:
    print(
        f"A chi-squared test of independence found a significant association between group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
else:
    print(
        f"A chi-squared test of independence showed no significant relationship between group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
