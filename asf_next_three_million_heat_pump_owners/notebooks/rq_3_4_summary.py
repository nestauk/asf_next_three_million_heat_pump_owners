# -*- coding: utf-8 -*-
# ---
# title: "RQs 3 & 4. Affordability and willingness to borrow"
# format:
#   revealjs:
#     scrollable: true
#     smaller: true
#     embed-resources: true
#     css: styles.css
#     slide-number: true
#     footer: "Identifying the next three million heat pump owners: YouGov Survey (RQs 3 & 4)"
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
# # RQ 3. To what extent, if at all, would installing a heat pump be affordable for UK households, without taking out additional borrowing?

# %% [markdown]
# ## Question that respondents were asked
#
# Upgrading from a fossil fuel boiler to an air source heat pump can cost £12,000 for a typical home. Often smaller, modern homes cost a bit less and larger older homes a bit more. In England, Wales and Scotland, grants are available of up to £7,500 to lower the cost of first-time heat pump installations...  **To what extent, if at all, would installing a heat pump be affordable for your household, without taking out additional borrowing?**

# %% [markdown]
# **Number of respondents**
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
# ## RQ 3. Overall

# %% [markdown]
# Most people said that installing a heat pump without taking out additional borrowing is unaffordable (72%) compared to 18% who said it was affordable and 10% who said they didn't know.

# %%
# Define response aggregation structure
q5a_response_aggregation_dict = {
    "Affordable (net)": [
        "Very affordable",
        "Fairly affordable",
    ],
    "Unaffordable (net)": [
        "Very unaffordable",
        "Fairly unaffordable",
    ],
}

# %%
# Summary dataframe
q5a_uk = summary.create_single_question_summary_frame(q5_base_data, "PEN_Q5Anew")

# Rename columns
q5a_uk.columns = ["Count", "Weighted count", "Percentage", "Weighted percentage"]

# Drop redundant rows
q5a_uk = q5a_uk.drop(index="Not asked").dropna().round(1)

# Combine categories
for combined_col, source_cols in q5a_response_aggregation_dict.items():
    q5a_uk.loc[combined_col, :] = q5a_uk.loc[source_cols, :].sum()

# Add confidence intervals for weighted proportions (95% ci)
weighted_pct_ci = summary.weighted_props_ci(q5_base_data, "PEN_Q5Anew", "weight") * 100
q5a_uk = q5a_uk.join(weighted_pct_ci[["Lower CI", "Upper CI", "SE"]])

# Add confidence intervals for combined weighted proportions
affordable_net = summary.weighted_props_ci_combine_categories(
    q5a_uk,
    "Very affordable",
    "Fairly affordable",
    "Affordable (net)",
)
q5a_uk.loc["Affordable (net)", "Lower CI"] = affordable_net["Lower CI"]
q5a_uk.loc["Affordable (net)", "Upper CI"] = affordable_net["Upper CI"]
q5a_uk.loc["Affordable (net)", "SE"] = affordable_net["SE"]

not_affordable_net = summary.weighted_props_ci_combine_categories(
    q5a_uk,
    "Very unaffordable",
    "Fairly unaffordable",
    "Unaffordable (net)",
)
q5a_uk.loc["Unaffordable (net)", "Lower CI"] = not_affordable_net["Lower CI"]
q5a_uk.loc["Unaffordable (net)", "Upper CI"] = not_affordable_net["Upper CI"]
q5a_uk.loc["Unaffordable (net)", "SE"] = not_affordable_net["SE"]

# Display only select columns
q5a_uk = q5a_uk.round(1)
q5a_uk[
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
    "To what extent, if at all, would installing a heat pump be affordable for your household, without taking out additional borrowing?"
)

# %% [markdown]
# ## RQ 3. By nation

# %% [markdown]
# The proportions of responses are similar across England, Scotland and Wales.

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
q5a_nations_df, q5a_nations_pct_df = summary.create_nations_summary_tables(
    q5_base_data, q5_base_data_wales, "PEN_Q5Anew", True
)
q5a_nations_pct_df = q5a_nations_pct_df.drop(["Northern Ireland", "UK"])

# %%
summary.create_stacked_horizontal_bar_chart(
    q5a_nations_pct_df, "Percentage", "Nation", 10, (10, 5)
).show()

# %%
# Compile 95% ci tables for each nation
weighted_props_ci_nations = {}
for nation, nation_data in q5_base_data_lookup.items():
    weighted_props_ci_nations[nation] = (
        summary.weighted_props_ci(nation_data, "PEN_Q5Anew", "weight") * 100
    ).round(1)

# Modify original nation comparison table with lower and upper CI values
q5a_nations_pct_df_ci = q5a_nations_pct_df.copy(deep=True)
q5a_nations_pct_df_ci = q5a_nations_pct_df_ci.astype(str)

for index in q5a_nations_pct_df_ci.index:
    for response in q5a_nations_pct_df_ci.columns:
        lower = weighted_props_ci_nations[index.lower()].loc[response, "Lower CI"]
        upper = weighted_props_ci_nations[index.lower()].loc[response, "Upper CI"]
        q5a_nations_pct_df_ci.loc[index, response] = (
            f"""{q5a_nations_pct_df_ci.loc[index, response]} ({lower} - {upper})"""
        )

q5a_nations_pct_df_ci = (
    q5a_nations_pct_df_ci.join(q5a_nations_df["Nation total"])
    .assign(n=lambda df: df["Nation total"].round(0).astype(int))
    .drop(columns="Nation total")
)

q5a_nations_pct_df_ci.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "To what extent, if at all, would installing a heat pump be affordable for your household, without taking out additional borrowing?"
)

# %% [markdown]
# ## RQ 1. By tenure

# %% [markdown]
# Most people across all tenure groups said that a heat pump is unaffordable.
#
# Only around a fifth of homeowners (21%) said that a heat pump is affordable without any additional borrowing.
#
# Homeowners who own outright have the highest proportion who said the it would be affordable (25%). Homeowners through part-ownership have the highest proportion who said that it would be unaffordable (90%).
#
# Interesting trend across tenure categories of proportions that said it would be unaffordable. Similar rates in owners and renters (73% and 74%), but then smaller proportions in other (54%) and neither (62%).
# - Note that the question was phrased "...affordable **for your household**" so respondents who were in one of the "Neither" or "Other" tenure groups were presumably answering from the point of view of the decision-makers in their household.

# %%
# Create Q5Anew response breakdown by tenure table for each nation
q5a_tenure_tables = {}
for group, base_data in q5_base_data_lookup.items():
    tables = {}
    counts_by_tenure, proportions_by_tenure = summary.generate_tenure_breakdown(
        base_data,
        "PEN_Q5Anew",
        q5a_response_aggregation_dict,
        proportions_axis=1,
    )
    tables["counts"] = counts_by_tenure
    tables["proportions"] = proportions_by_tenure

    q5a_tenure_tables[group] = tables

# %%
# Plot broad tenure categories comparison
q5a_tenure_pct_df = (
    q5a_tenure_tables["uk"]["proportions"]
    .reindex(index=["Own (net)", "Rent (net)", "Neither (net)", "Other"])
    .drop(columns=["Total", "Affordable (net)", "Unaffordable (net)"])
)

summary.create_stacked_horizontal_bar_chart(
    q5a_tenure_pct_df, "Percentage", "Nation", 10, (10, 5)
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

q5a_tenure_proportions_with_ci = summary.modify_proportion_crosstab_with_ci(
    base_data=q5_base_data,
    proportions_df=q5a_tenure_tables["uk"]["proportions"],
    aggregated_responses_column_dict=q5a_response_aggregation_dict,
    aggregated_responses_index_dict=aggregated_responses_tenure,
    index_survey_code="profile_house_tenure",
).drop(columns="Total")

# %%
# Display broad tenure categories for UK, proportions with 95% CI
q5a_tenure_proportions_with_ci_summary = (
    q5a_tenure_proportions_with_ci.join(q5a_tenure_tables["uk"]["counts"]["Total"])
    .assign(n=lambda df: df["Total"].round(0).astype(int))
    .drop(columns="Total")
)

q5a_tenure_proportions_with_ci_summary.drop(index="Total").style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "To what extent, if at all, would installing a heat pump be affordable for your household, without taking out additional borrowing?"
)

# %% [markdown]
# ## RQ 3. By household income

# %% [markdown]
# The proportion who said it would be affordable increases with increasing income brackets. It increases from 9% of those with household income of <£10k to 62% for >£150k.
#
# There is not as clear a trend looking at proportions who said it would be unaffordable.

# %%
# Create Q5A response breakdown by tenure table for each nation
q5a_income_tables = {}
for group, base_data in q5_base_data_lookup.items():
    tables = {}
    counts_by_income, proportions_by_income = summary.generate_income_breakdown(
        base_data,
        "PEN_Q5Anew",
        q5a_response_aggregation_dict,
        proportions_axis=1,
        aggregate_income=True,
    )
    tables["counts"] = counts_by_income
    tables["proportions"] = proportions_by_income

    q5a_income_tables[group] = tables

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

q5a_income_proportions_with_ci = summary.modify_proportion_crosstab_with_ci(
    base_data=q5_base_data,
    proportions_df=q5a_income_tables["uk"]["proportions"],
    aggregated_responses_column_dict=q5a_response_aggregation_dict,
    aggregated_responses_index_dict=aggregated_responses_income,
    index_survey_code="profile_gross_household",
).drop(columns="Total")

# %%
# Plot income bracket comparison table
q5a_income_pct_df = (
    q5a_income_tables["uk"]["proportions"]
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
    .drop(columns=["Total", "Affordable (net)", "Unaffordable (net)"])
)

summary.create_stacked_horizontal_bar_chart(
    q5a_income_pct_df, "Percentage", "Nation", 10, (10, 10)
).show()

# %% [markdown]
# Response breakdown by income bracket: Proportions of subpopulation (95% confidence interval) (%)

# %%
# Show only aggregated income brackets
q5a_income_proportions_with_ci_summary = (
    q5a_income_proportions_with_ci.reindex(
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
    .join(q5a_income_tables["uk"]["counts"]["Total"].round(0))
    .rename(columns={"Total": "n"})
    .assign(n=lambda df: df["n"].astype(int))
)

q5a_income_proportions_with_ci_summary.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "To what extent, if at all, would installing a heat pump be affordable for your household, without taking out additional borrowing?"
)

# %%
# Chi-squared contingency test across income brackets (net)
q5a_income_counts_summary = (
    q5a_income_tables["uk"]["counts"]
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
    .drop(columns=["Total", "Affordable (net)", "Unaffordable (net)"])
)
chi2_stat, p_val, dof, expected = chi2_contingency(q5a_income_counts_summary)

if p_val < 0.05:
    print(
        f"A chi-squared test of independence found a significant association between income group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
else:
    print(
        f"A chi-squared test of independence showed no significant relationship between income group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )

# %% [markdown]
# # RQ 4. How willing, if at all, would UK homeowners be to borrow funds from a mortgage provider/would you be to take out unsecured borrowing (e.g. via a personal loan or credit card) to fund the installation of low carbon heating?

# %% [markdown]
# ## Questions that respondents were asked
#
# - PEN_Q5Bnew. How willing, if at all, would you be to borrow funds from a mortgage provider to fund the installation of low carbon heating (such as heat pumps) in your home? (This could involve extending your existing mortgage or taking out a small additional mortgage)
# - PEN_Q5Cnew. How willing, if at all, would you be to take out unsecured borrowing (e.g. via a personal loan or credit card) in order to fund the installation of low carbon heating (such as heat pumps) in your home?

# %% [markdown]
# **Number of respondents**
# - Respondents who said they had a heat pump or are not homeowners were not asked.
# - From recoding free text responses, however, we identified three responses that indicated that the respondent has a heat pump. As these respondents should not have been asked, we have excluded them from the question base.
# - Base (unweighted): 4,507
# - Base (unweighted), after recoding: 4,506
# - Base (weighted), after recoding: 4,173

# %%
q5b_base_data = data[
    ~(data["PEN_Q2_3_recoded"] == True)
    & (
        (data["profile_house_tenure"] == "Own - outright")
        | (data["profile_house_tenure"] == "Own - with a mortgage")
        | (
            data["profile_house_tenure"]
            == "Own (part-own) - through shared ownership scheme (i.e. pay part mortgage, part rent)"
        )
    )
]

# %%
q5b_base_data_wales = wales_data[
    ~(wales_data["PEN_Q2_3_recoded"] == True)
    & (
        (wales_data["profile_house_tenure"] == "Own - outright")
        | (wales_data["profile_house_tenure"] == "Own - with a mortgage")
        | (
            wales_data["profile_house_tenure"]
            == "Own (part-own) - through shared ownership scheme (i.e. pay part mortgage, part rent)"
        )
    )
]

# %%
# Aggregated responses to PEN_Q5Bnew and PEN_Q5Cnew
q5b_q5c_response_aggregation_dict = {
    "Willing (net)": [
        "Very willing",
        "Fairly willing",
    ],
    "Not willing (net)": [
        "Not at all willing",
        "Not very willing",
    ],
}

# %% [markdown]
# ## RQ 4. Overall: Via mortgage provider

# %% [markdown]
# Most people (89%) are not willing to take out additional borrowing from a mortgage provider to fund the installation of low carbon heating system. Only 7% of people are willing.

# %%
# Summary dataframe
q5b_uk = summary.create_single_question_summary_frame(q5b_base_data, "PEN_Q5Bnew")

# Rename columns
q5b_uk.columns = ["Count", "Weighted count", "Percentage", "Weighted percentage"]

# Drop redundant rows
q5b_uk = q5b_uk.drop(index="Not asked").dropna().round(1)

# Combine categories
for combined_col, source_cols in q5b_q5c_response_aggregation_dict.items():
    q5b_uk.loc[combined_col, :] = q5b_uk.loc[source_cols, :].sum()

# Add confidence intervals for weighted proportions (95% ci)
weighted_pct_ci = summary.weighted_props_ci(q5b_base_data, "PEN_Q5Bnew", "weight") * 100
q5b_uk = q5b_uk.join(weighted_pct_ci[["Lower CI", "Upper CI", "SE"]])

# Add confidence intervals for combined weighted proportions
willing_net = summary.weighted_props_ci_combine_categories(
    q5b_uk,
    "Very willing",
    "Fairly willing",
    "Willing (net)",
)
q5b_uk.loc["Willing (net)", "Lower CI"] = willing_net["Lower CI"]
q5b_uk.loc["Willing (net)", "Upper CI"] = willing_net["Upper CI"]
q5b_uk.loc["Willing (net)", "SE"] = willing_net["SE"]

not_willing_net = summary.weighted_props_ci_combine_categories(
    q5b_uk,
    "Not at all willing",
    "Not very willing",
    "Not willing (net)",
)
q5b_uk.loc["Not willing (net)", "Lower CI"] = not_willing_net["Lower CI"]
q5b_uk.loc["Not willing (net)", "Upper CI"] = not_willing_net["Upper CI"]
q5b_uk.loc["Not willing (net)", "SE"] = not_willing_net["SE"]

# Display only select columns
q5b_uk = q5b_uk.round(1)
q5b_uk[
    ["Weighted count", "Weighted percentage", "Lower CI", "Upper CI", "SE"]
].style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "How willing, if at all, would you be to borrow funds from a mortgage provider to fund the installation of low carbon heating (such as heat pumps) in your home?"
).format(
    {
        "Weighted count": "{:.0f}",
        "Weighted percentage": "{:.1f}",
        "Lower CI": "{:.1f}",
        "Upper CI": "{:.1f}",
        "SE": "{:.1f}",
    }
)

# %% [markdown]
# ## RQ 4. Overall: Via unsecured borrowing

# %% [markdown]
# Most people (93%) are not willing to take out unsecured borrowing to fund the installation of low carbon heating system. This proportion is slightly higher than the proportion not willing to borrow via a mortgage provider (89%).
# Only 4% are willing, which is slightly lower than the proportion for borrowing via a mortgage provider (7%).

# %%
# Summary dataframe
q5c_uk = summary.create_single_question_summary_frame(q5b_base_data, "PEN_Q5Cnew")

# Rename columns
q5c_uk.columns = ["Count", "Weighted count", "Percentage", "Weighted percentage"]

# Drop redundant rows
q5c_uk = q5c_uk.drop(index="Not asked").dropna().round(1)

# Combine categories
for combined_col, source_cols in q5b_q5c_response_aggregation_dict.items():
    q5c_uk.loc[combined_col, :] = q5c_uk.loc[source_cols, :].sum()

# Add confidence intervals for weighted proportions (95% ci)
weighted_pct_ci = summary.weighted_props_ci(q5b_base_data, "PEN_Q5Bnew", "weight") * 100
q5c_uk = q5c_uk.join(weighted_pct_ci[["Lower CI", "Upper CI", "SE"]])

# Add confidence intervals for combined weighted proportions
willing_net = summary.weighted_props_ci_combine_categories(
    q5c_uk,
    "Very willing",
    "Fairly willing",
    "Willing (net)",
)
q5c_uk.loc["Willing (net)", "Lower CI"] = willing_net["Lower CI"]
q5c_uk.loc["Willing (net)", "Upper CI"] = willing_net["Upper CI"]
q5c_uk.loc["Willing (net)", "SE"] = willing_net["SE"]

not_willing_net = summary.weighted_props_ci_combine_categories(
    q5c_uk,
    "Not at all willing",
    "Not very willing",
    "Not willing (net)",
)
q5c_uk.loc["Not willing (net)", "Lower CI"] = not_willing_net["Lower CI"]
q5c_uk.loc["Not willing (net)", "Upper CI"] = not_willing_net["Upper CI"]
q5c_uk.loc["Not willing (net)", "SE"] = not_willing_net["SE"]

# Display only select columns
q5c_uk = q5c_uk.round(1)
q5c_uk[
    ["Weighted count", "Weighted percentage", "Lower CI", "Upper CI", "SE"]
].style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "How willing, if at all, would you be to take out unsecured borrowing (e.g. via a personal loan or credit card) in order to fund the installation of low carbon heating (such as heat pumps) in your home?"
).format(
    {
        "Weighted count": "{:.0f}",
        "Weighted percentage": "{:.1f}",
        "Lower CI": "{:.1f}",
        "Upper CI": "{:.1f}",
        "SE": "{:.1f}",
    }
)

# %% [markdown]
# ## RQ 4. By nation: Via mortgage provider

# %% [markdown]
# The proportions of responses are similar across England, Scotland and Wales, with the proportion of people who aren't willing being slightly higher in Wales (93% vs. 89% in England and 88% in Scotland).

# %%
# Individual nation datasets
q5b_base_data_england = q5b_base_data[q5b_base_data["gornewUK_6"] == True]
# q5b_base_data_wales = q5b_base_data[q5b_base_data["gornewUK_7"] == True]
q5b_base_data_scotland = q5b_base_data[q5b_base_data["gornewUK_8"] == True]
q5b_base_data_ni = q5b_base_data[q5b_base_data["gornewUK_9"] == True]

# Dataset lookup dictionary
q5b_base_data_lookup = {
    "uk": q5b_base_data,
    "england": q5b_base_data_england,
    "wales": q5b_base_data_wales,
    "scotland": q5b_base_data_scotland,
    "ni": q5b_base_data_ni,
}

# Create comparison table for all nations and UK
q5b_nations_df, q5b_nations_pct_df = summary.create_nations_summary_tables(
    q5b_base_data, q5b_base_data_wales, "PEN_Q5Bnew", True
)
q5b_nations_pct_df = q5b_nations_pct_df.drop("Northern Ireland")

# %%
summary.create_stacked_horizontal_bar_chart(
    q5b_nations_pct_df, "Percentage", "Nation", 10, (10, 5)
).show()

# %%
# Compile 95% ci tables for each nation
weighted_props_ci_nations = {}
for nation, nation_data in q5b_base_data_lookup.items():
    weighted_props_ci_nations[nation] = (
        summary.weighted_props_ci(nation_data, "PEN_Q5Bnew", "weight") * 100
    ).round(1)

# Modify original nation comparison table with lower and upper CI values
q5b_nations_pct_df_ci = q5b_nations_pct_df.copy(deep=True)
q5b_nations_pct_df_ci = q5b_nations_pct_df_ci.astype(str)

for index in q5b_nations_pct_df_ci.index:
    for response in q5b_nations_pct_df_ci.columns:
        lower = weighted_props_ci_nations[index.lower()].loc[response, "Lower CI"]
        upper = weighted_props_ci_nations[index.lower()].loc[response, "Upper CI"]
        q5b_nations_pct_df_ci.loc[index, response] = (
            f"""{q5b_nations_pct_df_ci.loc[index, response]} ({lower} - {upper})"""
        )

q5b_nations_pct_df_ci = (
    q5b_nations_pct_df_ci.join(q5b_nations_df["Nation total"])
    .assign(n=lambda df: df["Nation total"].round(0).astype(int))
    .drop(columns="Nation total")
)

q5b_nations_pct_df_ci.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "How willing, if at all, would you be to borrow funds from a mortgage provider to fund the installation of low carbon heating (such as heat pumps) in your home?"
)

# %% [markdown]
# ## RQ 4. By nation: Unsecured borrowing

# %% [markdown]
# The proportions of responses are similar across England, Scotland and Wales.

# %%
# Create comparison table for all nations and UK
q5c_nations_df, q5c_nations_pct_df = summary.create_nations_summary_tables(
    q5b_base_data, q5b_base_data_wales, "PEN_Q5Cnew", True
)
q5c_nations_pct_df = q5c_nations_pct_df.drop("Northern Ireland")

# %%
summary.create_stacked_horizontal_bar_chart(
    q5c_nations_pct_df, "Percentage", "Nation", 10, (10, 5)
).show()

# %% [markdown]
# Response breakdown by nation: Proportions of subpopulation (95% confidence interval) (%)

# %%
# Compile 95% ci tables for each nation
weighted_props_ci_nations = {}
for nation, nation_data in q5b_base_data_lookup.items():
    weighted_props_ci_nations[nation] = (
        summary.weighted_props_ci(nation_data, "PEN_Q5Cnew", "weight") * 100
    ).round(1)

# Modify original nation comparison table with lower and upper CI values
q5c_nations_pct_df_ci = q5c_nations_pct_df.copy(deep=True)
q5c_nations_pct_df_ci = q5c_nations_pct_df_ci.astype(str)

for index in q5c_nations_pct_df_ci.index:
    for response in q5c_nations_pct_df_ci.columns:
        lower = weighted_props_ci_nations[index.lower()].loc[response, "Lower CI"]
        upper = weighted_props_ci_nations[index.lower()].loc[response, "Upper CI"]
        q5c_nations_pct_df_ci.loc[index, response] = (
            f"""{q5c_nations_pct_df_ci.loc[index, response]} ({lower} - {upper})"""
        )

q5c_nations_pct_df_ci = (
    q5c_nations_pct_df_ci.join(q5b_nations_df["Nation total"])
    .assign(n=lambda df: df["Nation total"].round(0).astype(int))
    .drop(columns="Nation total")
)

q5c_nations_pct_df_ci.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "How willing, if at all, would you be to take out unsecured borrowing (e.g. via a personal loan or credit card) in order to fund the installation of low carbon heating (such as heat pumps) in your home?"
)

# %% [markdown]
# ## RQ 4. By household income: Via mortgage provider

# %% [markdown]
# Willingness to borrow funds via mortgage provider generally increases with income bracket. This proportion is 5% for those with household income under £10k and increases to 16% of those earning over £150k.

# %%
# Create Q5B response breakdown by tenure table for each nation
q5b_income_tables = {}
for group, base_data in q5b_base_data_lookup.items():
    tables = {}
    counts_by_income, proportions_by_income = summary.generate_income_breakdown(
        base_data,
        "PEN_Q5Bnew",
        q5b_q5c_response_aggregation_dict,
        proportions_axis=1,
        aggregate_income=True,
    )
    tables["counts"] = counts_by_income
    tables["proportions"] = proportions_by_income

    q5b_income_tables[group] = tables

# %%
# 95% confidence intervals for proportions
q5b_income_proportions_with_ci = summary.modify_proportion_crosstab_with_ci(
    base_data=q5b_base_data,
    proportions_df=q5b_income_tables["uk"]["proportions"],
    aggregated_responses_column_dict=q5b_q5c_response_aggregation_dict,
    aggregated_responses_index_dict=aggregated_responses_income,
    index_survey_code="profile_gross_household",
).drop(columns="Total")

# %%
# Plot income bracket comparison table
q5b_income_pct_df = (
    q5b_income_tables["uk"]["proportions"]
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
    .drop(columns=["Total", "Willing (net)", "Not willing (net)"])
)
summary.create_stacked_horizontal_bar_chart(
    q5b_income_pct_df, "Percentage", "Income", 10, (10, 10)
).show()

# %% [markdown]
# Response breakdown by income bracket: Proportions of subpopulation (95% confidence interval) (%)

# %%
# Show only aggregated income brackets
q5b_income_proportions_with_ci_summary = (
    q5b_income_proportions_with_ci.reindex(
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
    .join(q5a_income_tables["uk"]["counts"]["Total"].round(0))
    .rename(columns={"Total": "n"})
    .assign(n=lambda df: df["n"].astype(int))
)

q5b_income_proportions_with_ci_summary.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "How willing, if at all, would you be to borrow funds from a mortgage provider to fund the installation of low carbon heating (such as heat pumps) in your home?"
)

# %%
# Chi-squared contingency test across income brackets (net)
q5b_income_counts_summary = (
    q5b_income_tables["uk"]["counts"]
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
    .drop(columns=["Total", "Willing (net)", "Not willing (net)"])
)
chi2_stat, p_val, dof, expected = chi2_contingency(q5b_income_counts_summary)

if p_val < 0.05:
    print(
        f"A chi-squared test of independence found a significant association between income group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )
else:
    print(
        f"A chi-squared test of independence showed no significant relationship between income group subpopulations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
    )

# %% [markdown]
# ## RQ 4. By household income: Unsecured borrowing

# %% [markdown]
# Willingness to borrow funds via unsecured borrowing generally increases with income bracket. This proportion is 1% for those with household income under £10k and increases to 14% of those earning over £150k. These percentages are smaller in comparison to the mortgage provider question.

# %%
# Create Q5C response breakdown by tenure table for each nation
q5c_income_tables = {}
for group, base_data in q5b_base_data_lookup.items():
    tables = {}
    counts_by_income, proportions_by_income = summary.generate_income_breakdown(
        base_data,
        "PEN_Q5Cnew",
        q5b_q5c_response_aggregation_dict,
        proportions_axis=1,
        aggregate_income=True,
    )
    tables["counts"] = counts_by_income
    tables["proportions"] = proportions_by_income

    q5c_income_tables[group] = tables

# %%
# 95% confidence intervals for proportions
q5c_income_proportions_with_ci = summary.modify_proportion_crosstab_with_ci(
    base_data=q5b_base_data,
    proportions_df=q5c_income_tables["uk"]["proportions"],
    aggregated_responses_column_dict=q5b_q5c_response_aggregation_dict,
    aggregated_responses_index_dict=aggregated_responses_income,
    index_survey_code="profile_gross_household",
).drop(columns="Total")

# %%
# Plot income bracket comparison table
q5c_income_pct_df = (
    q5c_income_tables["uk"]["proportions"]
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
    .drop(columns=["Total", "Willing (net)", "Not willing (net)"])
)
summary.create_stacked_horizontal_bar_chart(
    q5c_income_pct_df, "Percentage", "Income bracket", 10, (10, 10)
).show()

# %%
# Show only aggregated income brackets
q5c_income_proportions_with_ci_summary = (
    q5c_income_proportions_with_ci.reindex(
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
    .join(q5a_income_tables["uk"]["counts"]["Total"].round(0))
    .rename(columns={"Total": "n"})
    .assign(n=lambda df: df["n"].astype(int))
)

q5c_income_proportions_with_ci_summary.style.set_table_styles(
    [
        {"selector": "td", "props": [("font-size", "18px")]},
        {"selector": "th", "props": [("font-size", "18px"), ("text-align", "left")]},
    ]
).set_caption(
    "How willing, if at all, would you be to take out unsecured borrowing (e.g. via a personal loan or credit card) in order to fund the installation of low carbon heating (such as heat pumps) in your home?"
)

# %%
# Exploratory

# %%
# # Affordability by age
# df = pd.crosstab(
#     index=q5_base_data["profile_julesage"],
#     columns=q5_base_data["PEN_Q5Anew"],
#     values=q5_base_data["weight"],
#     aggfunc="sum",
#     normalize="index",
# )
# df["Total"] = df.sum(axis=1)
# df = (df * 100).round(1)


# for combined_col, source_cols in q5a_response_aggregation_dict.items():
#     df[combined_col] = df[source_cols].sum(axis=1)

# df

# %%
# # Willingness to borrow (mortgage) by main benefit
# df = pd.crosstab(
#     index=q5_base_data["PEN_Q14"],
#     columns=q5_base_data["PEN_Q5Bnew"],
#     values=q5_base_data["weight"],
#     aggfunc="sum",
#     normalize="index",
# )
# df["Total"] = df.sum(axis=1)
# df = (df * 100).round(1)


# for combined_col, source_cols in q5b_q5c_response_aggregation_dict.items():
#     df[combined_col] = df[source_cols].sum(axis=1)

# df

# %%
# # Willingness to borrow (unsecured) by main benefit
# df = pd.crosstab(
#     index=q5_base_data["PEN_Q14"],
#     columns=q5_base_data["PEN_Q5Cnew"],
#     values=q5_base_data["weight"],
#     aggfunc="sum",
#     normalize="index",
# )
# df["Total"] = df.sum(axis=1)
# df = (df * 100).round(1)


# for combined_col, source_cols in q5b_q5c_response_aggregation_dict.items():
#     df[combined_col] = df[source_cols].sum(axis=1)

# df
