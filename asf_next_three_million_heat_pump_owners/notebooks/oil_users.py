# -*- coding: utf-8 -*-
# ---
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

# %% [markdown]
# ### Analysis on attitudes of oil users
# Date: 10/03/2026
# Aim: Generate statistics for oil users to compare with Segment D

# %%
import pandas as pd

from asf_next_three_million_heat_pump_owners import config
from asf_next_three_million_heat_pump_owners.utils import summary
from asf_next_three_million_heat_pump_owners.getters import load_data

# %%
# Load cleaned data
data, wales_data = load_data.get_analysis_ready_data()

# %%
# Recode Q5 to change responses of those that actually do have a heat pump from Not asked to Already have a heat pump
full_sample_data = data.copy(deep=True)
full_sample_data["PEN_Q5_recoded"] = full_sample_data.apply(
    lambda row: (
        "Already have a heat pump" if row["PEN_Q2_3_recoded"] == True else row["PEN_Q5"]
    ),
    axis=1,
)

# %%
# subset oil users
oil_group = full_sample_data[full_sample_data["PEN_Q2_5_recoded"] == True]

# %%
print(
    f"Number of respondents that heat with oil (weighted): {oil_group.groupby('PEN_Q2_5_recoded')['weight'].sum().round().iloc[0]}"
)

# %% [markdown]
# Distrust in the technology's performance Segment D exhibits a high level of distrust in heat pump technology. When surveyed, 45% of this group stated they are "not convinced that the technology is good enough to heat my home"
#
# PEN_Q11_2:
#     'Which, if any, of the following factors do you think would prevent you
#     from installing a heat pump in your home? (Please select all that apply. If there
#     is no specific barrier towards installing a heat pump in your home, please select
#     the "Not applicable" option): Not convinced that the technology is good enough
#     to heat my home'

# %%
filtered = oil_group[
    oil_group["PEN_Q11_2"] != "Not asked"
]  # respondents who are not homeowners, not asked

# %%
filtered.groupby("PEN_Q11_2")["weight"].sum()

# %%
filtered.groupby("PEN_Q11_2")["weight"].sum().sum()

# %%
filtered.groupby("PEN_Q11_2")["weight"].sum() / filtered["weight"].sum()

# %% [markdown]
# Concerns about property suitability Because Segment D typically lives in detached, rural properties, they often hold a strong belief that heat pumps are unsuited to their homes. Nearly a third (31%) cite "wouldn't be appropriate for my property" as a primary barrier to installation
#
# PEN_Q11_3:
#     'Which, if any, of the following factors do you think would prevent you
#     from installing a heat pump in your home? (Please select all that apply. If there
#     is no specific barrier towards installing a heat pump in your home, please select
#     the "Not applicable" option): Wouldn''t be appropriate for my property'

# %%
filtered = oil_group[
    oil_group["PEN_Q11_3"] != "Not asked"
]  # respondents who are not homeowners, not asked

# %%
filtered.groupby("PEN_Q11_3")["weight"].sum()

# %%
filtered.groupby("PEN_Q11_3")["weight"].sum().sum()

# %%
filtered.groupby("PEN_Q11_3")["weight"].sum() / filtered["weight"].sum()

# %%
summary.create_single_question_summary_frame(filtered, "house_type")

# %% [markdown]
# Satisfaction with their current system
# - 47% say they will stick with their current system until it physically needs replacing.
#
# PEN_Q11_14:
#     'Which, if any, of the following factors do you think would prevent
#     you from installing a heat pump in your home? (Please select all that apply. If
#     there is no specific barrier towards installing a heat pump in your home, please
#     select the "Not applicable" option): I will stick with my current system until
#     it needs replacing'
#
#
# - 44% state they are happy with their current heating system and do not see any reason to change it.
#
# PEN_Q11_12:
#     'Which, if any, of the following factors do you think would prevent
#     you from installing a heat pump in your home? (Please select all that apply. If
#     there is no specific barrier towards installing a heat pump in your home, please
#     select the "Not applicable" option): I am happy with my current heating system
#     and don''t see any reason to change it'

# %%
filtered = oil_group[
    oil_group["PEN_Q11_14"] != "Not asked"
]  # respondents who are not homeowners, not asked

# %%
filtered.groupby("PEN_Q11_14")["weight"].sum()

# %%
filtered.groupby("PEN_Q11_14")["weight"].sum().sum()

# %%
filtered.groupby("PEN_Q11_14")["weight"].sum() / filtered["weight"].sum()

# %%
filtered = oil_group[
    oil_group["PEN_Q11_12"] != "Not asked"
]  # respondents who are not homeowners, not asked

# %%
filtered.groupby("PEN_Q11_12")["weight"].sum()

# %%
filtered.groupby("PEN_Q11_12")["weight"].sum().sum()

# %%
filtered.groupby("PEN_Q11_12")["weight"].sum() / filtered["weight"].sum()

# %% [markdown]
# High upfront costs Cost remains a heavily cited deterrent, with participants noting that the "initial outlay is very costly". Interestingly, while the high upfront cost completely deters some from even looking into the technology, Segment D is actually slightly less likely than other groups to cite expense as their **primary barrier** (55% for Segment D compared to the 63% national average), likely because they generally have stable or higher incomes and own their homes outright.
#
# PEN_Q12:
#     Which ONE of the following factors do you think would prevent you the MOST
#     from installing a heat pump in your home?

# %%
filtered = oil_group[
    oil_group["PEN_Q12"] != "Not asked"
]  # respondents who are not homeowners, not asked

# %%
summary.create_single_question_summary_frame(filtered, "PEN_Q12")

# %% [markdown]
# Views on Subsidised Scenarios
# When this segment is explicitly presented with a scenario outlining a £12,000 heat pump installation supported by a £7,500 grant, the financial aid does make it viable for some, but it does not overcome the barrier for the majority:
# *   26% find it affordable: 7% consider this scenario "very affordable" and 19% consider it "fairly affordable"
# *   68% find it unaffordable: Despite the £7,500 grant, 41% still rate this as "very unaffordable" and 27% rate it as "fairly unaffordable".
#
# PEN_Q5Anew:
#     "Upgrading from a fossil fuel boiler to an air source heat pump can\
#     \ cost \xA312,000 for a typical home. Often smaller, modern homes cost a bit less\
#     \ and larger older homes a bit more. In England, Wales and Scotland, grants are\
#     \ available of up to \xA37,500 to lo"

# %%
filtered = oil_group[
    oil_group["PEN_Q5Anew"] != "Not asked"
]  # respondents already have heat pump, not asked

# %%
summary.create_single_question_summary_frame(filtered, "PEN_Q5Anew")

# %% [markdown]
# Extreme Reluctance to Finance or Borrow
# Segment D is almost universally opposed to taking on debt to pay for a heat pump.
# *   Unsecured Borrowing: They are highly resistant to using personal loans or credit cards to fund low-carbon heating, with 76% stating they are "not at all willing" and 16% "not very willing" to do so.
# *   Mortgage Extensions: There is similarly high resistance to using secured debt. 69% are "not at all willing" to extend an existing mortgage or take out a new one for a heat pump. This specific reluctance is highly logical given their lifestage; 81% of this segment own their homes. and the vast majority (67%) have already paid off their mortgages entirely.
# Ultimately, only 3% of this segment believe they would ever consider taking out a loan or paying in instalments specifically for a heat pump.
#
# PEN_Q5Bnew:
#     How willing, if at all, would you be to borrow funds from a mortgage
#     provider to fund the installation of low carbon heating (such as heat pumps) in
#     your home? (This could involve extending your existing mortgage or taking out
#     a small additional mortgage)
#   PEN_Q5Cnew:
#     How willing, if at all, would you be to take out unsecured borrowing
#     (e.g. via a personal loan or credit card) in order to fund the installation of
#     low carbon heating (such as heat pumps) in your home?

# %%
summary.create_single_question_summary_frame(oil_group, "profile_house_tenure")

# %%
filtered = oil_group[
    oil_group["PEN_Q5Bnew"] != "Not asked"
]  # respondents who are not homeowners, not asked

# %%
summary.create_single_question_summary_frame(filtered, "PEN_Q5Bnew")

# %%
summary.create_single_question_summary_frame(filtered, "profile_house_tenure")

# %%
filtered = oil_group[
    oil_group["PEN_Q5Cnew"] != "Not asked"
]  # respondents who are not homeowners, not asked

# %%
summary.create_single_question_summary_frame(filtered, "PEN_Q5Cnew")

# %% [markdown]
# *Additional*

# %% [markdown]
# PEN_Q4. Before taking this survey, had you ever heard of heat pumps as a home heating system? (Please select the option that best applies)

# %%
filtered = oil_group[
    oil_group["PEN_Q4"] != "Not asked"
]  # respondents already have a heat pump, not asked

# %%
summary.create_single_question_summary_frame(filtered, "PEN_Q4")

# %% [markdown]
# PEN_Q5: How likely, if at all, would you be to consider installing a heat pump in your home as your main heating source?

# %%
filtered = oil_group[
    oil_group["PEN_Q5"] != "Not asked"
]  # respondents already have a heat pump, not asked

# %%
summary.create_single_question_summary_frame(filtered, "PEN_Q5")

# %% [markdown]
# PEN_Q6: Before taking this survey, have you ever seen what a heat pump looks like?

# %%
filtered = oil_group[
    oil_group["PEN_Q6"] != "Not asked"
]  # respondents who are aware of heat pumps

# %%
summary.create_single_question_summary_frame(filtered, "PEN_Q6")

# %% [markdown]
# PEN_Q7: Have you ever been to a home that has a heat pump?

# %%
filtered = oil_group[
    oil_group["PEN_Q7"] != "Not asked"
]  # respondents who are aware of heat pumps

# %%
summary.create_single_question_summary_frame(filtered, "PEN_Q7")

# %% [markdown]
# PEN_Q8: Does anyone in your family, close friends or neighbours own a heat pump?

# %%
filtered = oil_group[
    oil_group["PEN_Q8"] != "Not asked"
]  # respondents who are aware of heat pumps

# %%
summary.create_single_question_summary_frame(filtered, "PEN_Q8")

# %% [markdown]
# PEN_Q9:
#     Have you ever had a conversation with your family, close friends or neighbours
#     about installing a heat pump in your home?

# %%
filtered = oil_group[
    oil_group["PEN_Q9"] != "Not asked"
]  # respondents who are aware of heat pumps

# %%
summary.create_single_question_summary_frame(filtered, "PEN_Q9")
