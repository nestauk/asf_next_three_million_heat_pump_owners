# -*- coding: utf-8 -*-
# ---
# title: "RQ 7. Perceived barriers and benefits"
# format:
#   revealjs:
#     scrollable: true
#     smaller: true
#     embed-resources: true
#     css: styles.css
#     slide-number: true
#     footer: "Identifying the next three million heat pump owners: YouGov Survey (RQ 7)"
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
# # RQ 7. What barriers and benefits do people foresee when it comes to installing a heat pump?

# %% [markdown]
# ## Questions that respondents were asked
#
# **Barriers**
# - PEN_Q11. Which, if any, of the following factors do you think would prevent you from installing a heat pump in your home? (Please select all that apply. If there is no specific barrier towards installing a heat pump in your home, please select the "Not applicable" option)
# - PEN_Q12. Which ONE of the following factors do you think would prevent you the MOST from installing a heat pump in your home
#
#
# **Benefits**
# - PEN_Q13. Which, if any, of the following benefits about heat pumps is likely to encourage you to install a heat pump in your home in the future? (Please select all that apply. If you would never get a heat pump installed in your home, please select the "Not applicable" option)
# - PEN_Q14. Which ONE of the following benefits about heat pumps is MOST likely to encourage you to install a heat pump in your home in the future?

# %% [markdown]
# ## Number of respondents
# - Respondents who said they had a heat pump or are not homeowners were not asked.
# - From recoding free text responses, however, we identified three responses that indicated that the respondent has a heat pump. As these respondents should not have been asked, we have excluded them from the question base.
# - Base (unweighted): 4,507
# - Base (unweighted), after recoding: 4,506
# - Base (weighted), after recoding: 4,173

# %%
q11_base_data = data[
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
q11_base_data_wales = wales_data[
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
# Individual nation datasets
q11_base_data_england = q11_base_data[q11_base_data["gornewUK_6"] == True]
# q11_base_data_wales = q11_base_data[q11_base_data["gornewUK_7"] == True]
q11_base_data_scotland = q11_base_data[q11_base_data["gornewUK_8"] == True]
q11_base_data_ni = q11_base_data[q11_base_data["gornewUK_9"] == True]

# Dataset lookup dictionary
q11_base_data_lookup = {
    "uk": q11_base_data,
    "england": q11_base_data_england,
    "wales": q11_base_data_wales,
    "scotland": q11_base_data_scotland,
    "ni": q11_base_data_ni,
}

# %% [markdown]
# ## RQ 7. UK and by nation: Barriers (Select all that apply)

# %% [markdown]
# The most common barrier across E/S/W and UK overall is "Too expensive to install", with 64% of people in the UK selecting this. The next most popular barriers that are not related to being happy with their current system are not being convinced by the technology (37%) and not thinking it makes financial sense as an investment (34%).
#
# A large portion of respondents selected barriers related to not seeing the need to change their current system (47% saying they will stick with the current system until it needs replacing, and 46% saying they are happy with their current system and see no reason to change).
#
# *The proportion of responses are similar across nations.*
#

# %%
barrier_options = [
    barrier for barrier in code_question_lookup.keys() if "Q11_" in barrier
]

# %%
multi_barrier_tables = {}
for barrier in barrier_options:
    if barrier == "PEN_Q11_collapsed":
        pass
    else:
        counts, proportions = summary.create_nations_summary_tables(
            q11_base_data, q11_base_data_wales, barrier, False
        )
        counts = counts.drop("Northern Ireland")
        proportions = proportions.drop("Northern Ireland")
        multi_barrier_tables[barrier] = {"counts": counts, "proportions": proportions}

# %%
# Summarise all barriers
multi_barrier_counts = pd.DataFrame()
multi_barrier_proportions = pd.DataFrame()
for barrier in barrier_options:
    if barrier == "PEN_Q11_collapsed":
        pass
    else:
        multi_barrier_counts = pd.concat(
            [multi_barrier_counts, multi_barrier_tables[barrier]["counts"]["True"]],
            axis=1,
        )
        multi_barrier_proportions = pd.concat(
            [
                multi_barrier_proportions,
                multi_barrier_tables[barrier]["proportions"]["True"],
            ],
            axis=1,
        )

barrier_column_names = []
for barrier in barrier_options:
    if barrier == "PEN_Q11_collapsed":
        pass
    else:
        barrier_column_names.append(
            f"{code_question_lookup.get(barrier).split(':')[1]}"
        )

for df in [multi_barrier_counts, multi_barrier_proportions]:
    df.columns = barrier_column_names

# %%
multi_barrier_df = multi_barrier_proportions.reset_index().rename(
    columns={"index": "Nation"}
)
multi_barrier_df_long = multi_barrier_df.melt(
    id_vars="Nation", var_name="Barrier", value_name="Percentage"
)

sorted_order = multi_barrier_df_long[
    multi_barrier_df_long["Nation"] == "England"
].sort_values("Percentage", ascending=False)["Barrier"]

plt.figure(figsize=(12, 10))
ax = sns.barplot(
    data=multi_barrier_df_long,
    y="Barrier",
    x="Percentage",
    hue="Nation",
    order=sorted_order,
)
ax.xaxis.grid(True, linestyle="-", alpha=0.3)
plt.xlabel("Percentage (%)")
plt.ylabel("Barrier")
plt.legend(title="Nation")
plt.tight_layout()
plt.show()

# %%
# Compile 95% ci tables for each barrier and each nation
barrier_proportions_ci = {}
for nation, nation_data in q11_base_data_lookup.items():
    nation_tables = {}
    for barrier in barrier_options:
        if barrier != "PEN_Q11_collapsed":
            nation_tables[barrier] = (
                summary.weighted_props_ci(nation_data, barrier, "weight") * 100
            ).round(1)
    barrier_proportions_ci[nation] = nation_tables

# Modify original nation comparison tables with lower and upper CI values
multi_barrier_proportions_with_ci = multi_barrier_proportions.copy(deep=True)
multi_barrier_proportions_with_ci = multi_barrier_proportions_with_ci.astype(str)

for barrier in barrier_options:
    if barrier != "PEN_Q11_collapsed":
        for index in multi_barrier_proportions_with_ci.index:
            lower = barrier_proportions_ci[index.lower()][barrier].loc[
                "True", "Lower CI"
            ]
            upper = barrier_proportions_ci[index.lower()][barrier].loc[
                "True", "Upper CI"
            ]

            multi_barrier_proportions_with_ci.loc[
                index, code_question_lookup.get(barrier).split(":")[1]
            ] = f"""{multi_barrier_proportions_with_ci.loc[index, code_question_lookup.get(barrier).split(':')[1]]} ({lower}-{upper})"""

# %%
multi_barrier_proportions_with_ci.T

# %%
# Chi-squared contingency test across nations
# Each barrier question

for barrier in barrier_options:
    if barrier != "PEN_Q11_collapsed":
        chi2_stat, p_val, dof, expected = chi2_contingency(
            multi_barrier_tables[barrier]["counts"].drop(
                index="UK", columns="Nation total"
            )
        )

        if p_val < 0.05:
            print(
                f"{barrier}: A chi-squared test of independence found a significant association between nations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
            )
        else:
            print(
                f"{barrier}: A chi-squared test of independence showed no significant relationship between nations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
            )

# %% [markdown]
# ## RQ 7. UK: Main barrier

# %% [markdown]
# The most common factor that would prevent people from installing a heat pump is expense (32% chose "Too expensive to install" as the factor that would prevent them the most). This is followed by being happy with their current system (13%) and not being convinced that technology can heat their home (12%).
#
#

# %%
# Summary dataframe
q12_uk = summary.create_single_question_summary_frame(q11_base_data, "PEN_Q12")

# Rename columns
q12_uk.columns = ["Count", "Weighted count", "Percentage", "Weighted percentage"]

# Drop redundant rows
q12_uk = q12_uk.drop(index="Not asked").dropna().round(1)

# Add confidence intervals for weighted proportions (95% ci)
weighted_pct_ci = summary.weighted_props_ci(q11_base_data, "PEN_Q12", "weight") * 100
q12_uk = q12_uk.join(weighted_pct_ci[["Lower CI", "Upper CI", "SE"]])

# Display only select columns
q12_uk = q12_uk.round(1)
q12_uk[["Weighted count", "Weighted percentage", "Lower CI", "Upper CI", "SE"]]

# %% [markdown]
# ## RQ 7. By nation: Main barrier

# %% [markdown]
# The top barriers are fairly similar across nations. Top three main barriers:
# - England: Too expensive, happy with current system, current system doesn't need replacing
# - Scotland: Too expensive, happy with current system, not appropriate for my property
# - Wales: Too expensive, not convinced of technology, happy with current system

# %%
# Create comparison table for all nations and UK
q12_nations_df, q12_nations_pct_df = summary.create_nations_summary_tables(
    q11_base_data, q11_base_data_wales, "PEN_Q12", True
)
q12_nations_pct_df = q12_nations_pct_df.drop("Northern Ireland")

# %%
# Add rank columns for comparing ranks across nations
q12_nations_pct_df_compare = q12_nations_pct_df.T.drop(columns="UK").sort_values(
    by="England", ascending=False
)
for nation in ["England", "Scotland", "Wales"]:
    q12_nations_pct_df_compare[f"{nation} rank"] = (
        q12_nations_pct_df_compare[nation].rank(ascending=False).astype(int)
    )

# %% [markdown]
# Response breakdown by nation: Proportions of subpopulation (95% confidence interval) (%)

# %%
# Compile 95% ci tables for each nation
weighted_props_ci_nations = {}
for nation, nation_data in q11_base_data_lookup.items():
    weighted_props_ci_nations[nation] = (
        summary.weighted_props_ci(nation_data, "PEN_Q12", "weight") * 100
    ).round(1)

# Modify original nation comparison table with lower and upper CI values
q12_nations_pct_df_ci = q12_nations_pct_df.copy(deep=True)
q12_nations_pct_df_ci = q12_nations_pct_df_ci.astype(str)

for index in q12_nations_pct_df_ci.index:
    for response in q12_nations_pct_df_ci.columns:
        lower = weighted_props_ci_nations[index.lower()].loc[response, "Lower CI"]
        upper = weighted_props_ci_nations[index.lower()].loc[response, "Upper CI"]
        q12_nations_pct_df_ci.loc[index, response] = (
            f"""{q12_nations_pct_df_ci.loc[index, response]} ({lower} - {upper})"""
        )

q12_nations_pct_df_ci = (
    q12_nations_pct_df_ci.join(q12_nations_df["Nation total"])
    .assign(**{"Nation total": lambda df: df["Nation total"].round(0).astype(int)})
    .rename(columns={"Nation total": "n"})
)

# %%
q12_nations_pct_df_ci.T.drop(columns="UK").join(
    q12_nations_pct_df_compare[["England rank", "Scotland rank", "Wales rank"]]
).sort_values(by="England rank", ascending=True)

# %% [markdown]
# ## RQ 7. UK and by nation: Benefits

# %% [markdown]
# The most commonly selected answer (33% of total UK) was that they would never get a heat pump. The next three most common answers were lowering energy bills (32%), environmental friendliness (29%) and independence from gas/oil/solid fuel (20%).
#
# The proportions across nations are similar.

# %%
benefit_options = [
    benefit for benefit in code_question_lookup.keys() if "Q13_" in benefit
]

# %%
multi_benefit_tables = {}
for benefit in benefit_options:
    if benefit == "PEN_Q13_collapsed":
        pass
    else:
        counts, proportions = summary.create_nations_summary_tables(
            q11_base_data, q11_base_data_wales, benefit, False
        )
        counts = counts.drop("Northern Ireland")
        proportions = proportions.drop("Northern Ireland")
        multi_benefit_tables[benefit] = {"counts": counts, "proportions": proportions}

# %%
# Summarise all benefits
multi_benefit_counts = pd.DataFrame()
multi_benefit_proportions = pd.DataFrame()
for benefit in benefit_options:
    if benefit == "PEN_Q13_collapsed":
        pass
    else:
        multi_benefit_counts = pd.concat(
            [multi_benefit_counts, multi_benefit_tables[benefit]["counts"]["True"]],
            axis=1,
        )
        multi_benefit_proportions = pd.concat(
            [
                multi_benefit_proportions,
                multi_benefit_tables[benefit]["proportions"]["True"],
            ],
            axis=1,
        )

barrier_column_names = []
for benefit in benefit_options:
    if benefit == "PEN_Q13_collapsed":
        pass
    else:
        barrier_column_names.append(
            f"{code_question_lookup.get(benefit).split(':')[1]}"
        )

for df in [multi_benefit_counts, multi_benefit_proportions]:
    df.columns = barrier_column_names

# %%
multi_benefit_df = multi_benefit_proportions.reset_index().rename(
    columns={"index": "Nation"}
)
multi_benefit_df_long = multi_benefit_df.melt(
    id_vars="Nation", var_name="Benefit", value_name="Percentage"
)

sorted_order = multi_benefit_df_long[
    multi_benefit_df_long["Nation"] == "England"
].sort_values("Percentage", ascending=False)["Benefit"]

plt.figure(figsize=(12, 6))
ax = sns.barplot(
    data=multi_benefit_df_long,
    y="Benefit",
    x="Percentage",
    hue="Nation",
    order=sorted_order,
)
ax.xaxis.grid(True, linestyle="-", alpha=0.3)
plt.xlabel("Percentage (%)")
plt.ylabel("Barrier")
plt.legend(title="Nation")
plt.tight_layout()
plt.show()

# %%
# Compile 95% ci tables for each benefit and each nation
benefit_proportions_ci = {}
for nation, nation_data in q11_base_data_lookup.items():
    nation_tables = {}
    for benefit in benefit_options:
        if benefit != "PEN_Q13_collapsed":
            nation_tables[benefit] = (
                summary.weighted_props_ci(nation_data, benefit, "weight") * 100
            ).round(1)
    benefit_proportions_ci[nation] = nation_tables

# Modify original nation comparison tables with lower and upper CI values
multi_benefit_proportions_with_ci = multi_benefit_proportions.copy(deep=True)
multi_benefit_proportions_with_ci = multi_benefit_proportions_with_ci.astype(str)

for benefit in benefit_options:
    if benefit != "PEN_Q13_collapsed":
        for index in multi_benefit_proportions_with_ci.index:
            try:
                lower = benefit_proportions_ci[index.lower()][benefit].loc[
                    "True", "Lower CI"
                ]
            except KeyError:
                lower = 0.0
            try:
                upper = benefit_proportions_ci[index.lower()][benefit].loc[
                    "True", "Upper CI"
                ]
            except KeyError:
                upper = 0.0

            multi_benefit_proportions_with_ci.loc[
                index, code_question_lookup.get(benefit).split(":")[1]
            ] = f"""{multi_benefit_proportions_with_ci.loc[index, code_question_lookup.get(benefit).split(':')[1]]} ({lower}-{upper})"""

# %%
multi_benefit_proportions_with_ci.T

# %%
# Chi-squared contingency test across nations
# Each benefit question

for benefit in benefit_options:
    if benefit != "PEN_Q13_collapsed":
        chi2_stat, p_val, dof, expected = chi2_contingency(
            multi_benefit_tables[benefit]["counts"].drop(
                index="UK", columns="Nation total"
            )
        )

        if p_val < 0.05:
            print(
                f"{benefit}: A chi-squared test of independence found a significant association between nations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
            )
        else:
            print(
                f"{benefit}: A chi-squared test of independence showed no significant relationship between nations and response (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.2e})."
            )

# %% [markdown]
# ## RQ 7. UK: Main benefit

# %% [markdown]
# The most common answer was that that they would never get a heat pump (33%). This is then followed by the benefits:
# - Can lower energy bills (18%)
# - More environmentally friendly (14%)
# - Independence from gas/oil/solid fuel (5%)

# %%
# Summary dataframe
q14_uk = summary.create_single_question_summary_frame(q11_base_data, "PEN_Q14")

# Rename columns
q14_uk.columns = ["Count", "Weighted count", "Percentage", "Weighted percentage"]

# Drop redundant rows
q14_uk = q14_uk.drop(index="Not asked").dropna().round(1)

# Add confidence intervals for weighted proportions (95% ci)
weighted_pct_ci = summary.weighted_props_ci(q11_base_data, "PEN_Q14", "weight") * 100
q14_uk = q14_uk.join(weighted_pct_ci[["Lower CI", "Upper CI", "SE"]])

# Display only select columns
q14_uk = q14_uk.round(1)
q14_uk[["Weighted count", "Weighted percentage", "Lower CI", "Upper CI", "SE"]]

# %% [markdown]
# ## RQ 7. By nation: Main benefit

# %% [markdown]
# Proportions are fairly similar across nations.
#
# The largest proportion for each nation (~33%) chose "Not applicable - I would never get a heat pump installed in my home"
#
# The ranking of top three main benefits for each nation is slightly different:
# - England: Lower energy bills, More environmentally friendly, Independence from gas/oil/solid fuel
# - Scotland: More environmentally friendly, Lower energy bills, Independence from gas/oil/solid fuel
# - Wales: Lower energy bills, More environmentally friendly, Independence from gas/oil/solid fuel
#
# "Don't know" was also a common answer across nations (12-15%) - which is a lot higher than the proportions who said "Don't know" when asked about the main barrier (1-4%).

# %%
# Create comparison table for all nations and UK
q14_nations_df, q14_nations_pct_df = summary.create_nations_summary_tables(
    q11_base_data, q11_base_data_wales, "PEN_Q14", True
)
q14_nations_pct_df = q14_nations_pct_df.drop("Northern Ireland")

# %%
# Add rank columns for comparing ranks across nations
q14_nations_pct_df_compare = q14_nations_pct_df.T.drop(columns="UK").sort_values(
    by="England", ascending=False
)
for nation in ["England", "Scotland", "Wales"]:
    q14_nations_pct_df_compare[f"{nation} rank"] = (
        q14_nations_pct_df_compare[nation].rank(ascending=False).astype(int)
    )

# %% [markdown]
# Response breakdown by nation: Proportions of subpopulation (95% confidence interval) (%)

# %%
# Compile 95% ci tables for each nation
weighted_props_ci_nations = {}
for nation, nation_data in q11_base_data_lookup.items():
    weighted_props_ci_nations[nation] = (
        summary.weighted_props_ci(nation_data, "PEN_Q14", "weight") * 100
    ).round(1)

# Modify original nation comparison table with lower and upper CI values
q14_nations_pct_df_ci = q14_nations_pct_df.copy(deep=True)
q14_nations_pct_df_ci = q14_nations_pct_df_ci.astype(str)

for index in q14_nations_pct_df_ci.index:
    for response in q14_nations_pct_df_ci.columns:
        lower = weighted_props_ci_nations[index.lower()].loc[response, "Lower CI"]
        upper = weighted_props_ci_nations[index.lower()].loc[response, "Upper CI"]
        q14_nations_pct_df_ci.loc[index, response] = (
            f"""{q14_nations_pct_df_ci.loc[index, response]} ({lower} - {upper})"""
        )

q14_nations_pct_df_ci = (
    q14_nations_pct_df_ci.join(q12_nations_df["Nation total"])
    .assign(**{"Nation total": lambda df: df["Nation total"].round(0).astype(int)})
    .rename(columns={"Nation total": "n"})
)

# %%
q14_nations_pct_df_ci.T.drop(columns="UK").join(
    q14_nations_pct_df_compare[["England rank", "Scotland rank", "Wales rank"]]
).sort_values(by="England rank", ascending=True)

# %% [markdown]
# ## RQ 7. Greater confidence on barriers compared to benefits

# %% [markdown]
# When asked about barriers or benefits that would prevent or encourage them from/to installing a heat pump in their home (select all factors questions), a much lower proportion of people said "Don't know" to barriers compared to benefits:
#
# - Barriers: 1.7% (1.3-2.1) selected "Don't know" (UK)
#
# - Benefits: 12.1% (11.1-13.0) selected "Don't know" (UK)
#
# This could suggest:
#
# - People are more confident identifying the barriers that would influence their decision-making rather than the benefits that would convince them
#
# - May be more important to focus on debunking misconceptions/lowering barriers
#
# - Knowledge gap about heat pump benefits and people may not have heard enough positive stories about heat pumps
#
# - There's still a lot of uncertainty and a lack of clarity on the positives of heat pumps

# %% [markdown]
# **The respondents who didn't know about benefits: What barriers did they select?**

# %%
# Respondents who said "Don't know" to benefits
indices_to_check = []
for i in q11_base_data[q11_base_data["PEN_Q13_977"] == "True"].index:
    indices_to_check.append(i)

# 539 respondents
df = (
    q11_base_data[q11_base_data["PEN_Q13_977"] == "True"]["PEN_Q12"]
    .value_counts()
    .to_frame(name="count")
)
df["percentage"] = ((df["count"] / df["count"].sum()) * 100).round(1)
df[["percentage"]]

# %% [markdown]
# - A third of respondents who said "Don't know" to what heat pump benefits would encourage them to install one in their home selected "Too expensive to install" when asked about the one barrier that would prevent them the most.
# - 90% of these respondents identified the biggest barrier for them (despite not identifying what the biggest encouragement for them would be). These respondents knew what would dissuade them, but didn't know what would persaude them.
# - Does this suggest that people are more likely to know what would deter them from doing something, but are less clear on what would persuade them?

# %%
# Exploratory section

# %%
# # Checking what main barrier respondents selected
# benefit = "Not applicable - I would never get a heat pump installed in my home"
# filtered_benefit_data = data[
#     ~(data["PEN_Q2_3_recoded"] == True)
#     & (data["PEN_Q14"] == benefit)
#     & (
#         (data["profile_house_tenure"] == "Own - outright")
#         | (data["profile_house_tenure"] == "Own - with a mortgage")
#         | (
#             data["profile_house_tenure"]
#             == "Own (part-own) - through shared ownership scheme (i.e. pay part mortgage, part rent)"
#         )
#     )
# ]
# summary.create_single_question_summary_frame(filtered_benefit_data, "PEN_Q12", "weight")

# %%
# # Checking what main benefit respondents selected
# barrier = "Too expensive to install"
# filtered_barrier_data = data[
#     ~(data["PEN_Q2_3_recoded"] == True)
#     & (data["PEN_Q12"] == barrier)
#     & (
#         (data["profile_house_tenure"] == "Own - outright")
#         | (data["profile_house_tenure"] == "Own - with a mortgage")
#         | (
#             data["profile_house_tenure"]
#             == "Own (part-own) - through shared ownership scheme (i.e. pay part mortgage, part rent)"
#         )
#     )
# ]
# summary.create_single_question_summary_frame(filtered_barrier_data, "PEN_Q14", "weight")

# %%
# # Main benefit by age
# df = pd.crosstab(
#     index=q11_base_data["profile_julesage"],
#     columns=q11_base_data["PEN_Q14"],
#     values=q11_base_data["weight"],
#     aggfunc="sum",
#     normalize="index",
# )
# df["Total"] = df.sum(axis=1)
# df = (df * 100).round(1)
# df.T

# %%
# # Main benefit by income
# counts_df, proportions_df = summary.generate_income_breakdown(q11_base_data, "PEN_Q14")
# proportions_df

# %%
# # Main barrier by age
# df = pd.crosstab(
#     index=q11_base_data["profile_julesage"],
#     columns=q11_base_data["PEN_Q12"],
#     values=q11_base_data["weight"],
#     aggfunc="sum",
#     normalize="index",
# )
# df["Total"] = df.sum(axis=1)
# df = (df * 100).round(1)
# df.T

# %%
# # Main barrier by income
# counts_df, proportions_df = summary.generate_income_breakdown(q11_base_data, "PEN_Q12")
# proportions_df
