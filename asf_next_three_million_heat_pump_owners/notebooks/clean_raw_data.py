# %%
import pandas as pd
from datetime import datetime
import yaml

from asf_next_three_million_heat_pump_owners import config
from asf_next_three_million_heat_pump_owners.utils import summary
from asf_next_three_million_heat_pump_owners.utils import cleaning
from asf_next_three_million_heat_pump_owners import PROJECT_DIR

# %% [markdown]
# ### Load raw data

# %%
data = pd.read_excel(
    config.get("raw_data_path"),
    header=[0, 1],
    sheet_name="SAV for Experian (Nesta Heat Pu",
    keep_default_na=False,
    na_values=[" "],
)

# %% [markdown]
# ### Raw data info

# %%
data.info()

# %% [markdown]
# ### Column cleaning and asserting data types
#
# The question code and full question are loaded as multiindex headers. For code brevity, the question code is kept as the column header and the full question is dropped. When the full question needs to be accessed, use the look-up dictionary `code_question_lookup`. The look-up dictionary is then stored in `config/base.yaml`.

# %%
# Create lookup dictionary for question code and full question
code_question_lookup = {col[0]: col[1] for col in data.columns}

# Remove second index (full question) of multi-index header
data.columns = data.columns.get_level_values(0)

# %% [markdown]
# #### Record number
# - Response type: integers [min: 0, max: 7024]
# - Number of NA: 0
# - Cleaning:
#     - Convert from int64 to int16

# %%
data["RecordNo"] = data["RecordNo"].astype("int16")

# %% [markdown]
# #### Weights
# - Response type: float [0.72397248856 to 6.67083832277]
# - Number of NA: 0
# - Cleaning:
#     - Convert to float

# %%
data["weight"] = data["weight"].astype(float)

# %% [markdown]
# #### Do you own or rent the home in which you live?
# - Response type: Text strings
# - Number of NA: 0
# - Cleaning:
#     - Replace encoded characters
#     - Convert to categorical

# %%
data["profile_house_tenure"] = data["profile_house_tenure"].replace(
    to_replace="â€“", value="-", regex=True
)

# %%
data["profile_house_tenure"] = pd.Categorical(
    data["profile_house_tenure"],
    categories=[
        "Own - outright",
        "Own - with a mortgage",
        "Own (part-own) - through shared ownership scheme (i.e. pay part mortgage, part rent)",
        "Rent - from a private landlord",
        "Rent - from my local authority",
        "Rent - from a housing association",
        "Neither - I live with my parents, family or friends but pay some rent to them",
        "Neither - I live rent-free with my parents, family or friends",
        "Other",
    ],
    ordered=False,
)

# %%
summary.create_single_question_summary_frame(data, "profile_house_tenure")

# %% [markdown]
# #### UK regions and countries lived in [calculated from postcode]
# - Response type: Text strings
# - Number of NA: 0
# - Cleaning:
#     - Convert to categorical

# %%
data["profile_GOR"] = pd.Categorical(
    data["profile_GOR"],
    categories=[
        "East Midlands",
        "East of England",
        "London",
        "North East",
        "North West",
        "South East",
        "South West",
        "West Midlands",
        "Yorkshire and the Humber",
        "Wales",
        "Scotland",
        "Northern Ireland",
    ],
    ordered=False,
)

# %%
summary.create_single_question_summary_frame(data, "profile_GOR")

# %% [markdown]
# #### How many people, including yourself, are there in your household? Please include both adults and children.
# - Response type: Integers or text strings
# - Number of NA: 0
# - Cleaning:
#     - Convert all integers to strings
#     - Convert to categorical

# %%
# Convert all to strings
data["profile_household_size"] = data["profile_household_size"].map(str)

# %%
data["profile_household_size"] = pd.Categorical(
    data["profile_household_size"],
    categories=[
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8 or more",
        "Don't know",
        "Prefer not to say",
    ],
    ordered=True,
)

# %%
summary.create_single_question_summary_frame(data, "profile_household_size")

# %% [markdown]
# #### What ethnic group best describes you? Please select one option only. (We ask the question in this way so that it is consistent with Census definitions.)
# - Response type: Text strings
# - Number of NA: 90 (Cannot identify whether they whether these were not asked based on a condition, or whether they were asked and did not give any answer)
# - Cleaning:
#     - Keep NaN as empty
#     - Convert to categorical

# %%
data["ethnicity_new"] = pd.Categorical(
    data["ethnicity_new"],
    categories=[
        "English / Welsh / Scottish / Northern Irish / British",  # White
        "Irish",  # White
        "Gypsy or Irish Traveller",  # White
        "Any other White background",  # White
        "White and Black Caribbean",  # Mixed / Multiple ethnic groups
        "White and Black African",  # Mixed / Multiple ethnic groups
        "White and Asian",  # Mixed / Multiple ethnic groups
        "Any other Mixed / Multiple ethnic background",  # Mixed / Multiple ethnic groups
        "Indian",  # Asian / Asian British
        "Pakistani",  # Asian / Asian British
        "Bangladeshi",  # Asian / Asian British
        "Chinese",  # Asian / Asian British
        "Any other Asian background",  # Asian / Asian British
        "African",  # Black / African / Caribbean / Black British
        "Caribbean",  # Black / African / Caribbean / Black British
        "Any other Black / African / Caribbean background",  # Black / African / Caribbean / Black British
        "Arab",  # Other ethnic group
        "Any other ethnic group",  # Other ethnic group
        "Prefer not to say",
    ],
    ordered=False,
)

# %%
summary.create_single_question_summary_frame(data, "ethnicity_new")

# %% [markdown]
# #### Gross HOUSEHOLD income is the combined income of all those earners in a household from all sources, including wages, salaries, or rents and before tax deductions. What is your gross household income?
# - Response type: Text strings
# - Number of NA: 0
# - Cleaning
#     - Replace encoded charactesr
#     - Convert to categorical

# %%
data["profile_gross_household"] = data["profile_gross_household"].replace(
    to_replace="Â", value="", regex=True
)

# %%
data["profile_gross_household"] = pd.Categorical(
    data["profile_gross_household"],
    categories=[
        "under £5,000 per year",
        "£5,000 to £9,999 per year",
        "£10,000 to £14,999 per year",
        "£15,000 to £19,999 per year",
        "£20,000 to £24,999 per year",
        "£25,000 to £29,999 per year",
        "£30,000 to £34,999 per year",
        "£35,000 to £39,999 per year",
        "£40,000 to £44,999 per year",
        "£45,000 to £49,999 per year",
        "£50,000 to £59,999 per year",
        "£60,000 to £69,999 per year",
        "£70,000 to £99,999 per year",
        "£100,000 to £149,999 per year",
        "£150,000 and over",
        "Don't know",
        "Prefer not to answer",
    ],
    ordered=True,
)

# %%
summary.create_single_question_summary_frame(data, "profile_gross_household")

# %% [markdown]
# #### What is the highest educational or work-related qualification you have?
# - Response type: Text strings
# - Number of NA: 0
# - Cleaning:
#     - Convert to categorical

# %%
data["profile_education_level"].value_counts(dropna=False)

# %%
data["profile_education_level"] = pd.Categorical(
    data["profile_education_level"],
    categories=[
        "No formal qualifications",
        "Youth training certificate/skillseekers",
        "Recognised trade apprenticeship completed",
        "Clerical and commercial",
        "City & Guilds certificate",
        "City & Guilds certificate - advanced",
        "ONC",
        "CSE grades 2-5",
        "CSE grade 1, GCE O level, GCSE, School Certificate",
        "Scottish Ordinary/ Lower Certificate",
        "GCE A level or Higher Certificate",
        "Scottish Higher Certificate",
        "Nursing qualification (e.g. SEN, SRN, SCM, RGN)",
        "Teaching qualification (not degree)",
        "University diploma",
        "University or CNAA first degree (e.g. BA, B.Sc, B.Ed)",
        "University or CNAA higher degree (e.g. M.Sc, Ph.D)",
        "Other technical, professional or higher qualification",
        "Don't know",
        "Prefer not to say",
    ],
    ordered=True,
)

# %%
summary.create_single_question_summary_frame(data, "profile_education_level")

# %% [markdown]
# #### Which, if any, of the following types of home best describes where you currently live?
# - Response type: Text strings
# - Number of NA: 0
# - Cleaning:
#     - Convert to categorical

# %%
data["house_type"] = pd.Categorical(
    data["house_type"],
    categories=[
        "Detached house",
        "Semi-detached house",
        "Terraced house",
        "Maisonette",
        "Studio/flat/apartment",
        "Bungalow",
        "Static Caravan/mobile home/trailer",
        "Other",
        "Don't know",
        "Prefer not to say",
    ],
    ordered=False,
)

# %%
summary.create_single_question_summary_frame(data, "house_type")

# %% [markdown]
# #### Thinking about your home (i.e. your main place of residence)...  Which of the following best describes the age of your house (i.e. when it was built)?
# - Response type: Text strings
# - Number of NA: 0
# - Cleaning:
#     - Convert to categorical

# %%
data["PEN_PropertyAge"] = pd.Categorical(
    data["PEN_PropertyAge"],
    categories=[
        "Pre 1920",
        "1920-1944",
        "1945-1964",
        "1965-1982",
        "1983-2002",
        "Post 2003",
        "Don't know",
    ],
    ordered=True,
)

# %%
summary.create_single_question_summary_frame(data, "PEN_PropertyAge")

# %% [markdown]
# #### How many bedrooms does your home have in total?
# - Response type: Integers or text strings
# - Number of NA: 0
# - Cleaning:
#     - Convert all to string
#     - Convert to categorical

# %%
# Convert all to strings
data["PEN_Bedrooms"] = data["PEN_Bedrooms"].map(str)

# %%
data["PEN_Bedrooms"] = pd.Categorical(
    data["PEN_Bedrooms"],
    categories=[
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6 or more",
        "Don't know",
    ],
    ordered=True,
)

# %%
summary.create_single_question_summary_frame(data, "PEN_Bedrooms")

# %% [markdown]
# #### Approximately, how large is the floor space in your home? (Please type your answer in square metres in the box below. If you are unsure, please give your best estimate)
#
# There are two column labels for this "PEN_Floorsize_AB" for the raw response, and "pen_floorsize_ab_range" which YouGov have collapsed floorsizes into ranges.
#
# "PEN_Floorsize_AB"
# - Response type: Number and text strings
# - Number of NA: 0
# - Cleaning:
#     - Convert to numeric, set DK to missing.

# %%
# Set DK to missing an convert to numeric data type
data["PEN_Floorsize_AB"] = pd.to_numeric(
    data["PEN_Floorsize_AB"], errors="coerce"
).astype("Int64")

# %% [markdown]
# "pen_floorsize_ab_range"
# - Response type: Text strings
# - Number of NA: 0
# - Cleaning:
#     - Replace "DK" with "Don't know"
#     - Convert to categorical

# %%
data["pen_floorsize_ab_range"] = data["pen_floorsize_ab_range"].replace(
    to_replace="DK", value="Don't know"
)

# %%
data["pen_floorsize_ab_range"] = pd.Categorical(
    data["pen_floorsize_ab_range"],
    categories=[
        "37 - 500",
        "500 - 1,000",
        "1,000 - 2,000",
        "2,000 - 3,000",
        "3,000 - 5,000",
        "5,000 - 7,000",
        "7,000 - 10,000",
        "Don't know",
    ],
    ordered=True,
)

# %%
summary.create_single_question_summary_frame(data, "pen_floorsize_ab_range")

# %% [markdown]
# #### Central heating is a central system that generates heat for multiple rooms… What type of central heating does your home have? (Please select all that apply. If there is no central heating in your home, please select the ‘Not applicable’ option)
# - For all columns
#     - Response type: Text strings
#     - Number of NA: 0
#     - Cleaning:
#         - Convert to boolean, except _open1 where convert to string
#         - Recode "Other" free text responses that should be classified in one of the given types
#
# - _1 Mains gas
# - _2 Tank or bottled gas
# - _3 Heat pump
# - _4 Electric (all others excluding heat pump, including storage heaters)
# - _5 Oil
# - _6 Wood (e.g. logs, waste wood or pellets)
# - _7 Solid fuel (e.g. coal)
# - _8 Renewable energy (e.g. solar thermal)
# - _9 District or communal heat network
# - _955 Other
# - _977 Don't know
# - _944 Not applicable - There is no central heating in my home
# - _open1 What type of central heating does you home have? -Other

# %%
# Modify full question labels in lookup
pen_q2_options = [option for option in code_question_lookup.keys() if "Q2_" in option]
for option in pen_q2_options:
    code_question_lookup[option] = (
        "Central heating is a central system that generates heat for multiple rooms… What type of central heating does your home have? (Please select all that apply. If there is no central heating in your home, please select the 'Not applicable' option): "
        + code_question_lookup.get(option)
    )

# %%
# Convert to boolean
for option in pen_q2_options:
    if option != "PEN_Q2_open1":
        data[option] = data[option].map({"Yes": True, "No": False})
    else:
        data[option] = data[option].map(str)

# %%
## Recoding free text responses

# Create recoded versions of each column
for option in pen_q2_options:
    if option != "PEN_Q2_open1":
        data[f"{option}_recoded"] = data[option].copy()

# %%
# Indices of responses that selected "Other" in addition to one of the specified options
other_plus_indices = data.index[
    data["PEN_Q2_955"]
    & (
        data["PEN_Q2_1"]
        | data["PEN_Q2_2"]
        | data["PEN_Q2_3"]
        | data["PEN_Q2_4"]
        | data["PEN_Q2_5"]
        | data["PEN_Q2_6"]
        | data["PEN_Q2_7"]
        | data["PEN_Q2_8"]
        | data["PEN_Q2_9"]
        | data["PEN_Q2_977"]
        | data["PEN_Q2_944"]
    )
]

# %%
## Invalid free text responses
invalid_responses_q2 = [
    "B",
    "Mi",
    "ÃŒ",
    "M",
    "O",
    "I",
    "Bh",
    "Up",
    "8",
    "6",
    "S",
    "Ppl",
    "Ok",
    "N",
]

invalid_responses_q2_indices = data[
    data["PEN_Q2_open1"].isin(invalid_responses_q2)
].index

# Create a new column for invalid responses
data["PEN_Q2_invalid_recoded"] = False

# If an invalid free response is given but a given option was selected, change their "Other" response to False
for i in invalid_responses_q2_indices:
    if i in other_plus_indices:
        data.loc[i, "PEN_Q2_955_recoded"] = False

# If an invalid free response is given and no other option was selected, change their "Other" response to False and add an "Invalid response" column with "True"
# Note all respondents with an invalid free text response selected another option so new "invalid" column is empty but keep code for future changes
for i in invalid_responses_q2_indices:
    if i not in other_plus_indices:
        data.loc[i, "PEN_Q2_955_recoded"] = False
        data.loc[i, "PEN_Q2_invalid_recoded"] = True  # Invalid

# %%
## Should be "Not applicable"

not_applicable_responses_q2 = [
    "These are studio flats and the only heat source I have is from the Aircon machine"
]
not_applicable_responses_q2_indices = data[
    data["PEN_Q2_open1"].isin(not_applicable_responses_q2)
].index

# No other option was selected, so change "Other" to False and "Not applicable - There is no central heating in my home" to True
for i in not_applicable_responses_q2_indices:
    data.loc[i, "PEN_Q2_955_recoded"] = False
    data.loc[i, "PEN_Q2_944_recoded"] = True  # Not applicable

# %%
## Unspecified fuel technologies - could be either gas or electric

unspec_fuel_responses_q2 = [
    "Underfloor heating",
    "Underfloor",
    "Central heating boiler",
    "A boiler",
    "The heating is powered by the boiler but I don't know if the boiler is powered by the gas or electricity.",
    "Under floor heating",
]

unspec_fuel_responses_q2_indices = data[
    data["PEN_Q2_open1"].isin(unspec_fuel_responses_q2)
].index

# If another given option was selected, assume that this was the known fuel source and change their "Other" option to False
for i in unspec_fuel_responses_q2_indices:
    if i in other_plus_indices:
        data.loc[i, "PEN_Q2_955_recoded"] = False

# If no other option was selected, change "Other" to False and assume they do not know fuel source so "Don't know" to True
for i in unspec_fuel_responses_q2_indices:
    if i not in other_plus_indices:
        data.loc[i, "PEN_Q2_955_recoded"] = False
        data.loc[i, "PEN_Q2_977_recoded"] = True  # Don't know

# %%
## District or communal heat network
communal_responses_q2 = ["Communal heating", "Communal"]
communal_responses_q2_indices = data[
    data["PEN_Q2_open1"].isin(communal_responses_q2)
].index

# Change "Other" option to False and ensure/change "District or communal heat network" to True
for i in communal_responses_q2_indices:
    data.loc[i, "PEN_Q2_955_recoded"] = False
    data.loc[i, "PEN_Q2_9_recoded"] = True  # District or communal heat network

# %%
## Wood
wood_responses_q2 = ["Biomass"]
wood_responses_q2_indices = data[data["PEN_Q2_open1"].isin(wood_responses_q2)].index

# Change "Other" option to False and ensure/change "Wood" to True
for i in wood_responses_q2_indices:
    data.loc[i, "PEN_Q2_955_recoded"] = False
    data.loc[i, "PEN_Q2_6_recoded"] = True  # Wood

# %%
## Heat pump responses
heat_pump_responses_q2 = [
    "Air source",
    "Hybrid boiler, gas and heat pump",
    "Air source pump",
]
heat_pump_responses_q2_indices = data[
    data["PEN_Q2_open1"].isin(heat_pump_responses_q2)
].index

# Change "Other" option to False and ensure/change "Heat pump" to True
for i in heat_pump_responses_q2_indices:
    data.loc[i, "PEN_Q2_955_recoded"] = False
    data.loc[i, "PEN_Q2_3_recoded"] = True  # Heat pump

# %%
## Electric responses
electric_responses_q2 = [
    "Electric",
    "Electric underfloor",
    "Dual fuel gas and electric",
    "Storage Heaters",
    "Infrared",
]
electric_responses_q2_indices = data[
    data["PEN_Q2_open1"].isin(electric_responses_q2)
].index

# Change "Other" option to False and ensure/change "Electric" to True
for i in electric_responses_q2_indices:
    data.loc[i, "PEN_Q2_955_recoded"] = False
    data.loc[i, "PEN_Q2_4_recoded"] = True  # Electric

# %%
## Gas responses (assume all refer to mains)
gas_responses_q2 = [
    "Hybrid boiler, gas and heat pump",
    "Gas warm air",
    "Back boiler",
    "Dual fuel gas and electric",
    "Gas fired central heating",
    "2 gas boilers",
    "Gas hot air",
    "Gas underfloor",
    "Combi boiler",  # Assume gas combi
    "Combi",  # Assume gas combi
    "combo boiler",  # Assume gas combi
    "Combiboiler",  # Assume gas combi
    "Hot air",  # Assume gas fired
    "Duct air",  # Assume gas fired
    "Warm air",  # Assume gas fired
]
gas_responses_q2_indices = data[data["PEN_Q2_open1"].isin(gas_responses_q2)].index

# Change "Other" option to False and ensure/change "Mains gas" to True
for i in gas_responses_q2_indices:
    data.loc[i, "PEN_Q2_955_recoded"] = False
    data.loc[i, "PEN_Q2_1_recoded"] = True  # Mains gas

# %%
## Tank or bottled gas responses
tank_gas_responses_q2 = ["LPG"]
tank_gas_responses_q2_indices = data[
    data["PEN_Q2_open1"].isin(tank_gas_responses_q2)
].index

# Change "Other" option to False and ensure/change "Tank or bottled gas" to True
for i in tank_gas_responses_q2_indices:
    data.loc[i, "PEN_Q2_955_recoded"] = False
    data.loc[i, "PEN_Q2_2_recoded"] = True  # Tank or bottled gas

# %%
## Renewable energy responses
renewable_responses_q2 = ["Electric PV"]
renewable_responses_q2_indices = data[
    data["PEN_Q2_open1"].isin(renewable_responses_q2)
].index

# Change "Other" option to False and ensure/change "Renewable energy" to True
for i in renewable_responses_q2_indices:
    data.loc[i, "PEN_Q2_955_recoded"] = False
    data.loc[i, "PEN_Q2_8_recoded"] = True  # Renewable energy

# %%
# Update question lookup for new recoded columns
for option in pen_q2_options:
    if option != "PEN_Q2_open1":
        recoded_col = f"{option}_recoded"
        code_question_lookup[recoded_col] = code_question_lookup.get(option)

code_question_lookup["PEN_Q2_invalid_recoded"] = (
    "Central heating is a central system that generates heat for multiple rooms… What type of central heating does your home have? (Please select all that apply. If there is no central heating in your home, please select the ‘Not applicable’ option): Invalid response"
)

# %%
## Collapse all responses into a single column
recoded_q2_options = [
    option
    for option in code_question_lookup.keys()
    if "PEN_Q2_" in option and "recoded" in option
]
q2_options_to_collapse = [
    option for option in recoded_q2_options if "open1" not in option
]
data = cleaning.collapse_select_all(
    data, q2_options_to_collapse, "PEN_Q2_recoded_collapsed", code_question_lookup
)

# Update question lookup for new collapsed recoded column
code_question_lookup["PEN_Q2_recoded_collapsed"] = (
    "Central heating is a central system that generates heat for multiple rooms… What type of central heating does your home have? (Please select all that apply. If there is no central heating in your home, please select the ‘Not applicable’ option)"
)

# %%
# Check changes from recoding for each option
q2_col_pairs = {}
for option in pen_q2_options:
    if option != "PEN_Q2_open1":
        q2_col_pairs[option] = f"{option}_recoded"

q2_col_pairs["PEN_Q2_invalid"] = "PEN_Q2_invalid_recoded"


count_comparison = {}
for original_col, recoded_col in q2_col_pairs.items():

    if original_col in data.columns and recoded_col in data.columns:
        counts_original = data[original_col].value_counts()
        counts_recoded = data[recoded_col].value_counts()
        count_comparison[recoded_col] = {
            "Original True": counts_original.get(True, 0),
            "Recoded True": counts_recoded.get(True, 0),
            "Original False": counts_original.get(False, 0),
            "Recoded False": counts_recoded.get(False, 0),
        }
    elif original_col not in data.columns:
        counts_recoded = data[recoded_col].value_counts()
        count_comparison[recoded_col] = {
            "Original True": 0,
            "Recoded True": counts_recoded.get(True, 0),
            "Original False": 0,
            "Recoded False": counts_recoded.get(False, 0),
        }

count_comparison_df = pd.DataFrame(count_comparison).T
count_comparison_df["Option"] = (
    count_comparison_df.index.map(code_question_lookup).str.split(":").str[1]
)

count_comparison_df

# %%
q2_recoded_summary = count_comparison_df[["Recoded True", "Recoded False"]].copy()
q2_recoded_summary["Percentage of respondents: True"] = (
    q2_recoded_summary["Recoded True"] / len(data)
) * 100
q2_recoded_summary["Percentage of respondents: False"] = (
    q2_recoded_summary["Recoded False"] / len(data)
) * 100
q2_recoded_summary[
    ["Percentage of respondents: True", "Percentage of respondents: False"]
] = q2_recoded_summary[
    ["Percentage of respondents: True", "Percentage of respondents: False"]
].round(
    2
)
q2_recoded_summary["Option"] = (
    count_comparison_df.index.map(code_question_lookup).str.split(":").str[1]
)

q2_recoded_summary

# %% [markdown]
# #### How old is your current central heating system?
# - Response type: Text strings
# - Number of NA: 170 (Not asked as they answered Not applicable to Q2 (do not have central heating))
# - Cleaning:
#     - Recode NaN to "Not asked"
#     - Convert to categorical

# %%
data["PEN_CurrentHeatingsystem_new"] = data["PEN_CurrentHeatingsystem_new"].fillna(
    "Not asked"
)

# %%
data["PEN_CurrentHeatingsystem_new"] = pd.Categorical(
    data["PEN_CurrentHeatingsystem_new"],
    categories=[
        "0-5 years",
        "6-10 years",
        "10-15 years",
        "More than 15 years",
        "Don't know",
        "Not asked",
    ],
    ordered=True,
)

# %%
summary.create_single_question_summary_frame(data, "PEN_CurrentHeatingsystem_new")

# %% [markdown]
# #### Still thinking about heating your home...  Approximately, for how long have you been using your current heating system?
# - Response type: Text strings
# - Number of NA: 170 (Not asked as they answered Not applicable to Q2 (do not have central heating))
# - Cleaning:
#     - Recode NaN to "Not asked"
#     - Convert to categorical

# %%
data["PEN_Heating"] = data["PEN_Heating"].fillna("Not asked")

# %%
data["PEN_Heating"] = pd.Categorical(
    data["PEN_Heating"],
    categories=[
        "0-5 years",
        "6-10 years",
        "10-15 years",
        "More than 15 years",
        "Don't know",
        "Not asked",
    ],
    ordered=True,
)

# %%
summary.create_single_question_summary_frame(data, "PEN_Heating")

# %% [markdown]
# #### EPC (Energy Performance Certificate) is a rating scheme that summarises the energy efficiency of homes, commercial and industrial properties... It uses a scale from 'A' (Very efficient) to 'G' (Inefficient) and the higher the rating, the more energy efficient your home is...  Before taking this survey, had you ever heard of EPC?
# - Response type: Text strings
# - Number of NA: 0
# - Cleaning:
#     - Convert to categorical

# %%
# Update lookup with full question
code_question_lookup["PEN_EPCawareness"] = (
    "EPC (Energy Performance Certificate) is a rating scheme that summarises the energy efficiency of homes, commercial and industrial properties... It uses a scale from 'A' (Very efficient) to 'G' (Inefficient) and the higher the rating, the more energy efficient your home is...  Before taking this survey, had you ever heard of EPC?"
)

# %%
data["PEN_EPCawareness"] = pd.Categorical(
    data["PEN_EPCawareness"],
    categories=[
        "Yes, I had",
        "No, I had not",
        "Don't know",
    ],
    ordered=True,
)

# %%
summary.create_single_question_summary_frame(data, "PEN_EPCawareness")

# %% [markdown]
# #### To the best of your knowledge, what is the current EPC grade of your home?
# - Response type: Text strings
# - Number of NA: 1893 (Not asked if they answered "No, I had not" (1391) or "Don't know" (502) to PEN_EPCawareness)
# - Cleaning:
#     - Convert NaN to "Not asked"
#     - Convert to categorical

# %%
data["PEN_EPCgradenew"] = data["PEN_EPCgradenew"].fillna("Not asked")

# %%
data["PEN_EPCgradenew"] = pd.Categorical(
    data["PEN_EPCgradenew"],
    categories=["A", "B", "C", "D", "E", "F", "G", "Don't know", "Not asked"],
    ordered=True,
)

# %%
summary.create_single_question_summary_frame(data, "PEN_EPCgradenew")

# %% [markdown]
# #### Does your home have a gas meter for 'mains gas'?
# - Response type: Text strings
# - Number of NA: 0
# - Cleaning:
#     - Convert to categorical

# %%
data["PEN_Q1"] = pd.Categorical(
    data["PEN_Q1"],
    categories=[
        "Yes, it does",
        "No, it does not",
        "Don't know",
    ],
    ordered=False,
)

# %%
summary.create_single_question_summary_frame(data, "PEN_Q1")

# %% [markdown]
# #### Which ONE of the following BEST describes the main type of central heating at your home?
# Respondents are asked to choose one from the options they selected in Q10 (PEN_Q2) if they chose more than 1.
#
# - Response type: Text strings
# - Number of NA: 0
# - Cleaning:
#     - Replace "$PENQ2open1" with "Other"
#     - Convert to categorical
#     - Create a recoded version of this question that matches the recoded PEN_Q2

# %%
data["PEN_Q2Anew"] = data["PEN_Q2Anew"].replace(to_replace="$PENQ2open1", value="Other")

# %%
## Other responses
other_responses_q2a_indices = data["PEN_Q2Anew"].index[data["PEN_Q2Anew"] == "Other"]

# Create new recoded version of column
data["PEN_Q2Anew_recoded"] = data["PEN_Q2Anew"]

# Recode responses if they were recoded in PEN_Q2
for i in other_responses_q2a_indices:
    if data["PEN_Q2_955_recoded"][i] == False:  # recoded to another option

        if data["PEN_Q2_1_recoded"][i] == True:
            data.loc[i, "PEN_Q2Anew_recoded"] = "Mains gas"

        if data["PEN_Q2_2_recoded"][i] == True:
            data.loc[i, "PEN_Q2Anew_recoded"] = "Tank or bottled gas"

        if data["PEN_Q2_3_recoded"][i] == True:
            data.loc[i, "PEN_Q2Anew_recoded"] = "Heat pump"

        if data["PEN_Q2_4_recoded"][i] == True:
            data.loc[i, "PEN_Q2Anew_recoded"] = (
                "Electric (all others excluding heat pump, including storage heaters)"
            )

        if data["PEN_Q2_5_recoded"][i] == True:
            data.loc[i, "PEN_Q2Anew_recoded"] = "Oil"

        if data["PEN_Q2_6_recoded"][i] == True:
            data.loc[i, "PEN_Q2Anew_recoded"] = (
                "Wood (e.g. logs, waste wood or pellets)"
            )

        if data["PEN_Q2_7_recoded"][i] == True:
            data.loc[i, "PEN_Q2Anew_recoded"] = "Solid fuel (e.g. coal)"

        if data["PEN_Q2_8_recoded"][i] == True:
            data.loc[i, "PEN_Q2Anew_recoded"] = "Renewable energy (e.g. solar thermal)"

        if data["PEN_Q2_9_recoded"][i] == True:
            data.loc[i, "PEN_Q2Anew_recoded"] = "District or communal heat network"

        if data["PEN_Q2_977_recoded"][i] == True:
            data.loc[i, "PEN_Q2Anew_recoded"] = "Don't know"

        if data["PEN_Q2_944_recoded"][i] == True:
            data.loc[i, "PEN_Q2Anew_recoded"] = (
                "Not applicable - There is no central heating in my home"
            )

# %%
data["PEN_Q2Anew"] = pd.Categorical(
    data["PEN_Q2Anew"],
    categories=[
        "Mains gas",
        "Tank or bottled gas",
        "Heat pump",
        "Electric (all others excluding heat pump, including storage heaters)",
        "Oil",
        "Wood (e.g. logs, waste wood or pellets)",
        "Solid fuel (e.g. coal)",
        "Renewable energy (e.g. solar thermal)",
        "District or communal heat network",
        "Other",
        "Don't know",
        "Not applicable - There is no central heating in my home",
    ],
    ordered=False,
)

# %%
summary.create_single_question_summary_frame(data, "PEN_Q2Anew")

# %%
data["PEN_Q2Anew_recoded"] = pd.Categorical(
    data["PEN_Q2Anew_recoded"],
    categories=[
        "Mains gas",
        "Tank or bottled gas",
        "Heat pump",
        "Electric (all others excluding heat pump, including storage heaters)",
        "Oil",
        "Wood (e.g. logs, waste wood or pellets)",
        "Solid fuel (e.g. coal)",
        "Renewable energy (e.g. solar thermal)",
        "District or communal heat network",
        "Other",
        "Don't know",
        "Not applicable - There is no central heating in my home",
    ],
    ordered=True,
)

# %%
summary.create_single_question_summary_frame(data, "PEN_Q2Anew_recoded")

# %% [markdown]
# #### Which, if any, of the following does your home have? (Please select all that apply)
# - For all columns
#     - Response type: Text strings
#     - Number of NA: 0
#     - Cleaning:
#         - Convert to boolean
#
# - _1 Private outdoor space
# - _2 Electric Vehicle (EV)
# - _3 An EV charger
# - _4 A hot water cylinder
# - _5 Space for a hot water cylinder
# - _6 Underfloor heating (not electric)
# - _7 An induction hob
# - _8 Solar panels
# - _9 Home battery
# - _10 A gas or oil boiler older than 10 years
# - _966 None of these
# - _977 Don't know
#

# %%
# Modify full question labels in lookup
pen_q3_options = [option for option in code_question_lookup.keys() if "Q3_" in option]
for option in pen_q3_options:
    code_question_lookup[option] = (
        "Which, if any, of the following does your home have? (Please select all that apply): "
        + code_question_lookup.get(option)
    )

# %%
# Convert to boolean
for option in pen_q3_options:
    data[option] = data[option].map({"Yes": True, "No": False})

# %%
# Collapse options to a single column
data = cleaning.collapse_select_all(
    data, pen_q3_options, "PEN_Q3_collapsed", code_question_lookup
)

# Update question lookup
code_question_lookup["PEN_Q3_collapsed"] = (
    "Which, if any, of the following does your home have? (Please select all that apply)"
)

# %%
summary_rows = []
for col in pen_q3_options:
    temp = (
        pd.Series(pd.Categorical(data[col], [True, False]), name=col)
        .to_frame()
        .assign(weight=data["weight"])
    )

    summary_rows.append(
        summary.create_single_question_summary_frame(temp, col)
        .reset_index()
        .melt(id_vars="index")
        .assign(
            column=lambda df: df["index"].astype(str).str.cat(df["variable"], sep="_")
        )
        .loc[:, ["column", "value"]]
        .set_index("column")
        .T.reset_index(drop=True)
        .rename_axis(None, axis=1)
        .rename(index={0: col})
    )

q3_counts_df = pd.concat(summary_rows)
q3_counts_df

# %% [markdown]
# #### Before taking this survey, had you ever heard of heat pumps as a home heating system? (Please select the option that best applies)
# - Response type: Text strings
# - Number of NA: 135 - Not asked because they selected "Heat pump" in Q2 (have a heat pump)
# - Cleaning:
#     - Change NaN to "Not asked"
#     - Convert to categorical

# %%
data["PEN_Q4"] = data["PEN_Q4"].fillna("Not asked")

# %%
data["PEN_Q4"] = pd.Categorical(
    data["PEN_Q4"],
    categories=[
        "I have heard of heat pumps, and I know what they are",
        "I have heard of heat pumps, but I don't know what they are",
        "I have never heard of heat pumps",
        "Not asked",
    ],
    ordered=False,
)

# %%
summary.create_single_question_summary_frame(data, "PEN_Q4")

# %% [markdown]
# #### Before taking this survey, have you ever seen what a heat pump looks like?
# - Response type: Text strings
# - Number of NA: 780 - Have a heat pump (Q2, n=135); Have never heard of heat pumps (Q4, n=645)
# - Cleaning:
#     - Change NaN to "Not asked"
#     - Convert to categorical

# %%
data["PEN_Q6"] = data["PEN_Q6"].fillna("Not asked")

# %%
data["PEN_Q6"] = pd.Categorical(
    data["PEN_Q6"],
    categories=[
        "Yes, I have",
        "No, I haven't",
        "Don't know/ can't recall",
        "Not asked",
    ],
    ordered=False,
)

# %%
summary.create_single_question_summary_frame(data, "PEN_Q6")

# %% [markdown]
# #### Have you ever been to a home that has a heat pump?
# - Response type: Text strings
# - Number of NA: 780 - Have a heat pump (Q2, n=135); Have never heard of heat pumps (Q4, n=645)
# - Cleaning:
#     - Change NaN to "Not asked"
#     - Convert to categorical

# %%
data["PEN_Q7"] = data["PEN_Q7"].fillna("Not asked")

# %%
data["PEN_Q7"] = pd.Categorical(
    data["PEN_Q7"],
    categories=[
        "Yes, I have",
        "No, I haven't",
        "Don't know/ can't recall",
        "Not asked",
    ],
    ordered=False,
)

# %%
summary.create_single_question_summary_frame(data, "PEN_Q7")

# %% [markdown]
# #### Does anyone in your family, close friends or neighbours own a heat pump?
# - Response type: Text strings
# - Number of NA: 780 - Have a heat pump (Q2, n=135); Have never heard of heat pumps (Q4, n=645)
# - Cleaning:
#     - Change NaN to "Not asked"
#     - Convert to categorical

# %%
data["PEN_Q8"] = data["PEN_Q8"].fillna("Not asked")

# %%
data["PEN_Q8"] = pd.Categorical(
    data["PEN_Q8"],
    categories=[
        "Yes, they do",
        "No, they do not",
        "Don't know",
        "Not asked",
    ],
    ordered=False,
)

# %%
summary.create_single_question_summary_frame(data, "PEN_Q8")

# %% [markdown]
# #### Have you ever had a conversation with your family, close friends or neighbours about installing a heat pump in your home?
# - Response type: Text strings
# - Number of NA: 780 - Have a heat pump (Q2, n=135); Have never heard of heat pumps (Q4, n=645)
# - Cleaning:
#     - Change NaN to "Not asked"
#     - Convert to categorical

# %%
data["PEN_Q9"] = data["PEN_Q9"].fillna("Not asked")

# %%
data["PEN_Q9"] = pd.Categorical(
    data["PEN_Q9"],
    categories=[
        "Yes, I have",
        "No, I haven't",
        "Don't know",
        "Not asked",
    ],
    ordered=False,
)

# %%
summary.create_single_question_summary_frame(data, "PEN_Q9")

# %% [markdown]
# #### For the following questions, please imagine that in the next 5 years...(i.e. from now till February 2030), you want to install a central heating system for your home/ your current heating system needs replacing...A heat pump can heat a home. An outside unit (see picture) takes heat from the air, concentrates it, and puts it into radiators. It's like a fridge in reverse. It works effectively in all weather conditions, including on cold days. Heat pumps run on electricity so differ to boilers, which run on fossil fuels (gas or oil). This means that heat pumps can run using electricity that is generated from renewable resources, such as wind or sun, which do not produce carbon emissions...How likely, if at all, would you be to consider installing a heat pump in your home as your main heating source? (Please select the option that best applies. Even if you are not currently a homeowner, we are still interested in your opinion.)
# - Response type: Text strings
# - Number of NA: 135 - Not asked because they selected "Heat pump" in Q2 (have a heat pump)
# - Cleaning:
#     - Change NaN to "Not asked"
#     - Convert to categorical

# %%
data["PEN_Q5"] = data["PEN_Q5"].fillna("Not asked")

# %%
data["PEN_Q5"] = pd.Categorical(
    data["PEN_Q5"],
    categories=[
        "Definitely would consider",
        "Probably would consider",
        "Probably wouldn't consider",
        "Definitely wouldn't consider",
        "Don't know",
        "Not asked",
    ],
    ordered=True,
)

# %%
summary.create_single_question_summary_frame(data, "PEN_Q5")

# %% [markdown]
# #### Upgrading from a fossil fuel boiler to an air source heat pump can cost £12,000 for a typical home. Often smaller, modern homes cost a bit less and larger older homes a bit more. In England, Wales and Scotland, grants are available of up to £7,500 to lower the cost of first-time heat pump installations...  To what extent, if at all, would installing a heat pump be affordable for your household, without taking out additional borrowing?
# - Response type: Text strings
# - Number of NA: 135 - Not asked because they selected "Heat pump" in Q2 (have a heat pump)
# - Cleaning:
#     - Change NaN to "Not asked"
#     - Convert to categorical

# %%
data["PEN_Q5Anew"] = data["PEN_Q5Anew"].fillna("Not asked")

# %%
data["PEN_Q5Anew"] = pd.Categorical(
    data["PEN_Q5Anew"],
    categories=[
        "Very affordable",
        "Fairly affordable",
        "Fairly unaffordable",
        "Very unaffordable",
        "Don't know",
        "Not asked",
    ],
    ordered=True,
)

# %%
summary.create_single_question_summary_frame(data, "PEN_Q5Anew")

# %% [markdown]
# #### How willing, if at all, would you be to borrow funds from a mortgage provider to fund the installation of low carbon heating (such as heat pumps) in your home? (This could involve extending your existing mortgage or taking out a small additional mortgage)
# - Response type: Text strings
# - Number of NA: 2518 - those who do not own a heat pump (did not select Heat pump in Q2) AND are homeowners (selected one of the "Own" options in profile_house_tenure)
# - Cleaning:
#     - Change NaN to "Not asked"
#     - Convert to categorical

# %%
data["PEN_Q5Bnew"] = data["PEN_Q5Bnew"].fillna("Not asked")

# %%
data["PEN_Q5Bnew"] = pd.Categorical(
    data["PEN_Q5Bnew"],
    categories=[
        "Very willing",
        "Fairly willing",
        "Not very willing",
        "Not at all willing",
        "Don't know",
        "Not asked",
    ],
    ordered=True,
)

# %%
summary.create_single_question_summary_frame(data, "PEN_Q5Bnew")

# %% [markdown]
# #### How willing, if at all, would you be to take out unsecured borrowing (e.g. via a personal loan or credit card) in order to fund the installation of low carbon heating (such as heat pumps) in your home?
# - Response type: Text strings
# - Number of NA: 2518 - those who do not own a heat pump (did not select Heat pump in Q2) AND are homeowners (selected one of the "Own" options in profile_house_tenure)
# - Cleaning:
#     - Change NaN to "Not asked"
#     - Convert to categorical

# %%
data["PEN_Q5Cnew"] = data["PEN_Q5Cnew"].fillna("Not asked")

# %%
data["PEN_Q5Cnew"] = pd.Categorical(
    data["PEN_Q5Cnew"],
    categories=[
        "Very willing",
        "Fairly willing",
        "Not very willing",
        "Not at all willing",
        "Don't know",
        "Not asked",
    ],
    ordered=True,
)

# %%
summary.create_single_question_summary_frame(data, "PEN_Q5Cnew")

# %% [markdown]
# #### Which, if any, of the following factors do you think would prevent you from installing a heat pump in your home? (Please select all that apply. If there is no specific barrier towards installing a heat pump in your home, please select the "Not applicable" option)
# - For all columns
#     - Response type: Text strings
#     - Number of NA: 2518 - those who do not own a heat pump (did not select Heat pump in Q2) AND are homeowners (selected one of the "Own" options in profile_house_tenure)
#     - Cleaning:
#         - Convert "Yes"/"False" to "True"/"False" (strings)
#         - Replace NaN with "Not asked" (string)
#
# - _1 Too expensive to install
# - _2 Not convinced that the technology is good enough to heat my home
# - _3 Wouldn't be appropriate for my property
# - _4 I worry I may end up paying more for my energy bills
# - _5 Difficulty finding a trader I trust to install it
# - _6 I don't have enough information to make an informed decision
# - _7 Too much hassle to research and find an installer
# - _8 Too much disruption to my home
# - _9 Planning permission
# - _10 I'm not the decision-maker about energy in my household
# - _11 Others in my household would need to be convinced
# - _12 I am happy with my current heating system and don't see any reason to change it
# - _13 I am planning to move home in the near future
# - _14 I will stick with my current system until it needs replacing
# - _15 I don't think it makes financial sense as an investment
# - _977 Don't know
# - _944 Not applicable - There is no specific barrier towards installing a heat pump in my home
#

# %%
# Modify full question labels in lookup
pen_q11_options = [option for option in code_question_lookup.keys() if "Q11_" in option]
for option in pen_q11_options:
    code_question_lookup[option] = (
        """Which, if any, of the following factors do you think would prevent you from installing a heat pump in your home? (Please select all that apply. If there is no specific barrier towards installing a heat pump in your home, please select the "Not applicable" option): """
        + code_question_lookup.get(option)
    )

# %%
# Convert to boolean and replace NaN
for option in pen_q11_options:
    data[option] = data[option].map({"Yes": "True", "No": "False"})
    data[option] = data[option].fillna("Not asked")

# %%
# Collapse options to a single column
data = cleaning.collapse_select_all(
    data, pen_q11_options, "PEN_Q11_collapsed", code_question_lookup
)

# Update question lookup
code_question_lookup["PEN_Q11_collapsed"] = (
    """Which, if any, of the following factors do you think would prevent you from installing a heat pump in your home? (Please select all that apply. If there is no specific barrier towards installing a heat pump in your home, please select the "Not applicable" option)"""
)

# %% [markdown]
# #### Which ONE of the following factors do you think would prevent you the MOST from installing a heat pump in your home?
# Respondents asked to choose one option from PEN_Q11 if selected more than one.
# - Response type: Text strings
# - Number of NA: 2518 - those who do not own a heat pump (did not select Heat pump in Q2) AND are homeowners (selected one of the "Own" options in profile_house_tenure)
# - Cleaning:
#     - Replace NaN with "Not asked"
#     - Convert to categorical

# %%
data["PEN_Q12"] = data["PEN_Q12"].fillna("Not asked")

# %%
data["PEN_Q12"] = pd.Categorical(
    data["PEN_Q12"],
    categories=[
        "Too expensive to install",
        "Not convinced that the technology is good enough to heat my home",
        "Wouldn't be appropriate for my property",
        "I worry I may end up paying more for my energy bills",
        "Difficulty finding a trader I trust to install it",
        "I don't have enough information to make an informed decision",
        "Too much hassle to research and find an installer",
        "Too much disruption to my home",
        "Planning permission",
        "I'm not the decision-maker about energy in my household",
        "Others in my household would need to be convinced",
        "I am happy with my current heating system and don't see any reason to change it",
        "I am planning to move home in the near future",
        "I will stick with my current system until it needs replacing",
        "I don't think it makes financial sense as an investment",
        "Don't know",
        "Not applicable - There is no specific barrier towards installing a heat pump in my home",
        "Not asked",
    ],
    ordered=False,
)

# %%
summary.create_single_question_summary_frame(data, "PEN_Q12")

# %% [markdown]
# #### PEN_Q13. Which, if any, of the following benefits about heat pumps is likely to encourage you to install a heat pump in your home in the future? (Please select all that apply. If you would never get a heat pump installed in your home, please select the "Not applicable" option)
# - For all columns
#     - Response type: Text strings
#     - Number of NA: 2518 - those who do not own a heat pump (did not select Heat pump in Q2) AND are homeowners (selected one of the "Own" options in profile_house_tenure)
#     - Cleaning:
#         - Convert "Yes"/"False" to "True"/"False" (string)
#         - Replace NaN with "Not asked" (string)
#
# - _1 It can help lower household energy bills
# - _2 It's more environmentally friendly and sustainable than traditional heating systems
# - _3 To future proof my home as it will be one of the main ways people heat their home in the future
# - _4 It has a long lifespan and needs little maintenance
# - _5 It would heat my home better/more reliably than my current system
# - _6 Independence from gas/oil/solid fuel
# - _7 Other people I know are doing it and I don't want to get left behind
# - _8 To use the electricity  generated by solar panels
# - _977 Don't know
# - _944 Not applicable - I would never get a heat pump installed in my home

# %%
# Modify full question labels in lookup
pen_q13_options = [option for option in code_question_lookup.keys() if "Q13_" in option]
for option in pen_q13_options:
    code_question_lookup[option] = (
        'Which, if any, of the following benefits about heat pumps is likely to encourage you to install a heat pump in your home in the future? (Please select all that apply. If you would never get a heat pump installed in your home, please select the "Not applicable" option): '
        + code_question_lookup.get(option)
    )

# %%
# Convert to boolean
for option in pen_q13_options:
    data[option] = data[option].map({"Yes": "True", "No": "False"})
    data[option] = data[option].fillna("Not asked")

# %%
# Collapse options to a single column
data = cleaning.collapse_select_all(
    data, pen_q13_options, "PEN_Q13_collapsed", code_question_lookup
)

# Update question lookup
code_question_lookup["PEN_Q13_collapsed"] = (
    """Which, if any, of the following benefits about heat pumps is likely to encourage you to install a heat pump in your home in the future? (Please select all that apply. If you would never get a heat pump installed in your home, please select the "Not applicable" option)"""
)

# %% [markdown]
# #### Which ONE of the following benefits about heat pumps is MOST likely to encourage you to install a heat pump in your home in the future?
# Respondents asked to choose one option from PEN_Q13 if selected more than one.
# - For all columns
#     - Response type: Text strings
#     - Number of NA: 2518 - those who do not own a heat pump (did not select Heat pump in Q2) AND are homeowners (selected one of the "Own" options in profile_house_tenure)
#     - Cleaning:
#         - Replace NaN with "Not asked"
#         - Convert to categorical

# %%
data["PEN_Q14"] = data["PEN_Q14"].fillna("Not asked")

# %%
data["PEN_Q14"] = pd.Categorical(
    data["PEN_Q14"],
    categories=[
        "It can help lower household energy bills",
        "It's more environmentally friendly and sustainable than traditional heating systems",
        "To future proof my home as it will be one of the main ways people heat their home in the future",
        "It has a long lifespan and needs little maintenance",
        "It would heat my home better/more reliably than my current system",
        "Independence from gas/oil/solid fuel",
        "Other people I know are doing it and I don't want to get left behind",
        "To use the electricity  generated by solar panels",
        "Don't know",
        "Not applicable - I would never get a heat pump installed in my home",
        "Not asked",
    ],
    ordered=False,
)

# %%
summary.create_single_question_summary_frame(data, "PEN_Q14")

# %% [markdown]
# ### PEN_Q15. For the following question, please imagine that you find out that your boiler will stop working in a year...  Which, if any, of the following sources would you consult in order to figure out what you should do? (Please select all that apply. If you would not consult any sources, please select the 'Not applicable' option)
# - For all columns
#     - Response type: Text strings
#     - Number of NA: 3249 - Homeowners AND have a mains gas/tank or bottled gas as main type of central heating not asked
#     - Cleaning:
#         - Convert all open responses to string
#         - Replace NaN to "Not asked" (string)
#         - Convert to "Yes"/"No" to "True"/"False" (string)
#         - Recode "Other" free text responses that should be classified in one of the given types
#
# - _1 Household members
# - _2 Wider family
# - _3 Friends
# - _4 Boiler engineer
# - _5 Social media
# - _6 Energy company
# - _7 Internet searches
# - _8 AI/ ChatGPT
# - _955 Other
# - _977 Don't know
# - _944 Not applicable - I would not consult any sources
# - _open1 Which, if any, of the following sources would you consult in order to figure out what you should do? -Other

# %%
# Modify full question labels in lookup
pen_q15_options = [option for option in code_question_lookup.keys() if "Q15_" in option]
for option in pen_q15_options:
    code_question_lookup[option] = (
        "For the following question, please imagine that you find out that your boiler will stop working in a year...  Which, if any, of the following sources would you consult in order to figure out what you should do? (Please select all that apply. If you would not consult any sources, please select the 'Not applicable' option): "
        + code_question_lookup.get(option)
    )

# %%
# Convert to boolean and replace NaN
for option in pen_q15_options:
    data[option] = data[option].map(str)
    if option != "PEN_Q15_open1":
        data[option] = data[option].map({"Yes": "True", "No": "False"})
        data[option] = data[option].fillna("Not asked")

# %%
## Recoding free text responses

# Create recoded versions of each column
for option in pen_q15_options:
    if option != "PEN_Q15_open1":
        data[f"{option}_recoded"] = data[option].copy()

# %%
# Indices of responses that selected "Other" in addition to one of the specified options
other_plus_q15_indices = data.index[
    (data["PEN_Q15_955"] == "True")
    & (
        (data["PEN_Q15_1"] == "True")
        | (data["PEN_Q15_2"] == "True")
        | (data["PEN_Q15_3"] == "True")
        | (data["PEN_Q15_4"] == "True")
        | (data["PEN_Q15_5"] == "True")
        | (data["PEN_Q15_6"] == "True")
        | (data["PEN_Q15_7"] == "True")
        | (data["PEN_Q15_8"] == "True")
        | (data["PEN_Q15_977"] == "True")
        | (data["PEN_Q15_944"] == "True")
    )
]

# %%
## Invalid free text responses (NEW)
invalid_responses_q15 = [
    "I",
    ".",
    "N",
    "H",
    "J",
    "Iâ€™m",
    "Det",
    "9",
]
invalid_responses_q15_indices = data[
    data["PEN_Q15_open1"].isin(invalid_responses_q15)
].index

# Create a new column for invalid responses
data["PEN_Q15_invalid_recoded"] = data["PEN_Q15_955"].apply(
    lambda x: x if x == "Not asked" else "False"
)

# If an invalid free response is given but a given option was selected, change their "Other" response to False
for i in invalid_responses_q15_indices:
    if i in other_plus_q15_indices:
        data.loc[i, "PEN_Q15_955_recoded"] = "False"

# If an invalid free response is given and no other option was selected, change their "Other" response to False and add an "Invalid response" column with "True"
for i in invalid_responses_q15_indices:
    if i not in other_plus_q15_indices:
        data.loc[i, "PEN_Q15_955_recoded"] = "False"
        data.loc[i, "PEN_Q15_invalid_recoded"] = "True"

# %%
## Government responses (NEW)
government_responses_q15_indices = data[
    data["PEN_Q15_open1"].str.contains("government", case=False, na=False)
]["PEN_Q15_open1"].index

# Create a new column for government responses
data["PEN_Q15_government_recoded"] = data["PEN_Q15_955"].apply(
    lambda x: x if x == "Not asked" else "False"
)

# If an invalid free response is given but a given option was selected, change their "Other" response to False
for i in government_responses_q15_indices:
    if i in other_plus_q15_indices:
        data.loc[i, "PEN_Q15_955_recoded"] = "False"

# If an invalid free response is given and no other option was selected, change their "Other" response to False and add an "Invalid response" column with "True"
for i in government_responses_q15_indices:
    if i not in other_plus_q15_indices:
        data.loc[i, "PEN_Q15_955_recoded"] = "False"
        data.loc[i, "PEN_Q15_government_recoded"] = "True"

# %%
## Should be "Wider family"
wider_family_responses_q15 = ["My dad", "Son", "My son"]
wider_family_responses_q15_indices = data[
    data["PEN_Q15_open1"].isin(wider_family_responses_q15)
].index

# Change "Other" option to False and ensure/change "Wider family" to True
for i in wider_family_responses_q15_indices:
    data.loc[i, "PEN_Q15_955_recoded"] = "False"
    data.loc[i, "PEN_Q15_2_recoded"] = "True"  # Wider family

# %%
## Should be "Boiler engineer"
boiler_engineer_responses_q15 = [
    "the installer as it's still insured (7 months old)",
    "Iâ€™m gas save engineer",
    "Son is boiler engineer",
    "Boiler instilation companies",
    "I am a retired engineer. I have always installed and serviced my own boilers.",
]
boiler_engineer_responses_q15_indices = data[
    data["PEN_Q15_open1"].isin(boiler_engineer_responses_q15)
].index

# Change "Other" option to False and ensure/change "Boiler engineer" to True
for i in boiler_engineer_responses_q15_indices:
    data.loc[i, "PEN_Q15_955_recoded"] = "False"
    data.loc[i, "PEN_Q15_4_recoded"] = "True"  # Boiler engineer

# %%
## Should be "Energy company"
energy_company_responses_q15 = [
    "British gas",
    "Why arent energy companies funding research training and education around alternate home heating systems? They have profited for decades from huge domestic gas/electric bills they should be at the forefront pf the transition to renewable energy sources",
    "British gas as we have a service agreement",
    "British Gas",
    "british gas",
]
energy_company_responses_q15_indices = data[
    data["PEN_Q15_open1"].isin(energy_company_responses_q15)
].index

# Change "Other" option to False and ensure/change "Energy company" to True
for i in energy_company_responses_q15_indices:
    data.loc[i, "PEN_Q15_955_recoded"] = "False"
    data.loc[i, "PEN_Q15_6_recoded"] = "True"  # Energy company

# %%
## Should be "Internet searches"
internet_searches_responses_q15 = ["Green websites", "Online reviews & articles"]
internet_searches_responses_q15_indices = data[
    data["PEN_Q15_open1"].isin(internet_searches_responses_q15)
].index

# Change "Other" option to False and ensure/change "Internet searches" to True
for i in internet_searches_responses_q15_indices:
    data.loc[i, "PEN_Q15_955_recoded"] = "False"
    data.loc[i, "PEN_Q15_7_recoded"] = "True"  # Internet searches

# %%
## Should be "Not applicable"
not_applicable_responses_q15 = [
    "I would decide based on current knowledge",
    "I wouldn't need to consult any sources as I would replace my boiler, like for like",
    "I would rather buy a new boiler to replace mine. If you want me to get a heat pump you pay for it all. Not a loan or subsidise you pay the full cost.",
]
not_applicable_responses_q15_indices = data[
    data["PEN_Q15_open1"].isin(not_applicable_responses_q15)
].index

# Change "Other" option to False and ensure/change "Not applicable" to True
for i in not_applicable_responses_q15_indices:
    data.loc[i, "PEN_Q15_955_recoded"] = "False"
    data.loc[i, "PEN_Q15_944_recoded"] = (
        "True "  # Not applicable - I would not consult any sources
    )

# %%
# Update question lookup for new recoded columns
for option in pen_q15_options:
    if option != "PEN_Q15_open1":
        recoded_col = f"{option}_recoded"
        code_question_lookup[recoded_col] = code_question_lookup.get(option)

code_question_lookup["PEN_Q15_invalid_recoded"] = (
    """For the following question, please imagine that you find out that your boiler will stop working in a year...  Which, if any, of the following sources would you consult in order to figure out what you should do? (Please select all that apply. If you would not consult any sources, please select the 'Not applicable' option): Invalid response"""
)
code_question_lookup["PEN_Q15_government_recoded"] = (
    """For the following question, please imagine that you find out that your boiler will stop working in a year...  Which, if any, of the following sources would you consult in order to figure out what you should do? (Please select all that apply. If you would not consult any sources, please select the 'Not applicable' option): Government"""
)

# %%
## Collapse all responses into a single column
recoded_q15_options = [
    option
    for option in code_question_lookup.keys()
    if "PEN_Q15_" in option and "recoded" in option
]
q15_options_to_collapse = [
    option for option in recoded_q15_options if "open1" not in option
]
data = cleaning.collapse_select_all(
    data, q15_options_to_collapse, "PEN_Q15_recoded_collapsed", code_question_lookup
)

# Update question lookup for new collapsed recoded column
code_question_lookup["PEN_Q15_recoded_collapsed"] = (
    """For the following question, please imagine that you find out that your boiler will stop working in a year...  Which, if any, of the following sources would you consult in order to figure out what you should do? (Please select all that apply. If you would not consult any sources, please select the 'Not applicable' option)"""
)

# %%
# Check changes from recoding for each option
q15_col_pairs = {}
for option in pen_q15_options:
    if option != "PEN_Q15_open1":
        q15_col_pairs[option] = f"{option}_recoded"

q15_col_pairs["PEN_Q15_invalid"] = "PEN_Q15_invalid_recoded"
q15_col_pairs["PEN_Q15_government"] = "PEN_Q15_government_recoded"


count_comparison = {}
for original_col, recoded_col in q15_col_pairs.items():

    if original_col in data.columns and recoded_col in data.columns:
        counts_original = data[original_col].value_counts()
        counts_recoded = data[recoded_col].value_counts()
        count_comparison[recoded_col] = {
            "Original True": counts_original.get("True", 0),
            "Recoded True": counts_recoded.get("True", 0),
            "Original False": counts_original.get("False", 0),
            "Recoded False": counts_recoded.get("False", 0),
        }
    elif original_col not in data.columns:
        counts_recoded = data[recoded_col].value_counts()
        count_comparison[recoded_col] = {
            "Original True": 0,
            "Recoded True": counts_recoded.get("True", 0),
            "Original False": 0,
            "Recoded False": counts_recoded.get("False", 0),
        }

count_comparison_df = pd.DataFrame(count_comparison).T
count_comparison_df

# %% [markdown]
# #### PEN_Q10new. Which, if any, of the following have you ever had an online or in person quote/ estimate for? (Please select all that apply)
# - For all columns
#     - Response type: Text strings
#     - Number of NA: 0
#     - Cleaning:
#         - Convert to boolean
#
# - _1 Heat pump
# - _2 Solar panels
# - _3 EV charger
# - _966 None of these
# - _977 Don't know/can't recall
#

# %%
# Modify full question labels in lookup
pen_q10_options = [
    option for option in code_question_lookup.keys() if "Q10new_" in option
]
for option in pen_q10_options:
    code_question_lookup[option] = (
        "Which, if any, of the following have you ever had an online or in person quote/ estimate for? (Please select all that apply): "
        + code_question_lookup.get(option)
    )

# %%
# Convert to boolean
for option in pen_q10_options:
    data[option] = data[option].map({"Yes": True, "No": False}).astype(bool)

# %%
# Collapse to single column
data = cleaning.collapse_select_all(
    data, pen_q10_options, "PEN_Q10new_collapsed", code_question_lookup
)

# Update question lookup
code_question_lookup["PEN_Q10new_collapsed"] = (
    "Which, if any, of the following have you ever had an online or in person quote/ estimate for? (Please select all that apply)"
)

# %%
summary_rows = []
for col in pen_q10_options:
    temp = (
        pd.Series(pd.Categorical(data[col], [True, False]), name=col)
        .to_frame()
        .assign(weight=data["weight"])
    )

    summary_rows.append(
        summary.create_single_question_summary_frame(temp, col)
        .reset_index()
        .melt(id_vars="index")
        .assign(
            column=lambda df: df["index"].astype(str).str.cat(df["variable"], sep="_")
        )
        .loc[:, ["column", "value"]]
        .set_index("column")
        .T.reset_index(drop=True)
        .rename_axis(None, axis=1)
        .rename(index={0: col})
    )

q10_counts_df = pd.concat(summary_rows)
q10_counts_df

# %% [markdown]
# #### On a different topic...  Which ONE of the following statements BEST describes your feelings towards new technologies?
# - Response type: Text strings
# - Number of NA: 0
# - Cleaning:
#     - Convert to categorical

# %%
data["PEN_Q19"] = pd.Categorical(
    data["PEN_Q19"],
    categories=[
        "I often try new technology",
        "I am open to trying new technology but I don't rush into it",
        "I am concerned about trying new technology",
        "Other",
        "Don't know",
    ],
    ordered=False,
)

# %%
summary.create_single_question_summary_frame(data, "PEN_Q19")

# %% [markdown]
# Open responses to PEN_Q19
# - Response type: Text strings and integers
# - Number of NA: 0
# - Cleaning:
#     - Convert to string

# %%
# Free text responses (144 unique responses)
data["PEN_Q19_open1"] = data["PEN_Q19_open1"].map(str)

# %% [markdown]
# #### PEN_Q20. We would now like to understand how you've used loans and financing previously...  Which, if any, of the following have you ever taken out a loan to pay for, or paid in instalments? (Please select all that apply)
# - For all columns
#     - Response type: Text strings
#     - Number of NA: 0
#     - Cleaning:
#         - Convert to boolean
#
# - _1 A mortgage
# - _2 A car
# - _3 Kitchen improvements
# - _4 Bathroom improvements
# - _5 Smaller purchases (such as clothes or phones)
# - _6 Home energy improvements excluding heat pumps (such as insulation, new windows, or solar panels)
# - _7 A heat pump
# - _8 Home extensions
# - _966 None of these
# - _977 Don't know/can't recall
#

# %%
# Modify full question labels in lookup
pen_q20_options = [option for option in code_question_lookup.keys() if "Q20_" in option]
for option in pen_q20_options:
    code_question_lookup[option] = (
        "We would now like to understand how you've used loans and financing previously...  Which, if any, of the following have you ever taken out a loan to pay for, or paid in instalments? (Please select all that apply): "
        + code_question_lookup.get(option)
    )

# %%
# Convert to boolean
for option in pen_q20_options:
    data[option] = data[option].map({"Yes": True, "No": False}).astype(bool)

# %%
# Collapse options to a single column
data = cleaning.collapse_select_all(
    data, pen_q20_options, "PEN_Q20_collapsed", code_question_lookup
)

# Update question lookup
code_question_lookup["PEN_Q20_collapsed"] = (
    "We would now like to understand how you've used loans and financing previously...  Which, if any, of the following have you ever taken out a loan to pay for, or paid in instalments? (Please select all that apply)"
)

# %% [markdown]
# #### PEN_Q21. Thinking about the future...  For which, if any, of the following do you think you will ever take out a loan to pay or pay any of the following in instalments? (Please select all that apply)
# - For all columns
#     - Response type: Text strings
#     - Number of NA: 0
#     - Cleaning:
#         - Convert to boolean
#
# - _1 A mortgage
# - _2 A car
# - _3 Kitchen improvements
# - _4 Bathroom improvements
# - _5 Smaller purchases (such as clothes or phones)
# - _6 Home energy improvements excluding heat pumps (such as insulation, new windows, or solar panels)
# - _7 A heat pump
# - _8 Home extensions
# - _966 None of these
# - _977 Don't know/can't recall
#

# %%
# Modify full question labels in lookup
pen_q21_options = [option for option in code_question_lookup.keys() if "Q21_" in option]
for option in pen_q21_options:
    code_question_lookup[option] = (
        "Thinking about the future...  For which, if any, of the following do you think you will ever take out a loan to pay or pay any of the following in instalments? (Please select all that apply): "
        + code_question_lookup.get(option)
    )

# %%
# Convert to boolean
for option in pen_q21_options:
    data[option] = data[option].map({"Yes": True, "No": False}).astype(bool)

# %%
# Collapse options to a single column
data = cleaning.collapse_select_all(
    data, pen_q21_options, "PEN_Q21_collapsed", code_question_lookup
)

# Update question lookup
code_question_lookup["PEN_Q21_collapsed"] = (
    "Thinking about the future...  For which, if any, of the following do you think you will ever take out a loan to pay or pay any of the following in instalments? (Please select all that apply)"
)

# %% [markdown]
# #### How desirable, if at all, would buying a house that has a heat pump be for you?
# - Response type: Text strings
# - Number of NA: 0
# - Cleaning:
#     - Convert to categorical

# %%
data["PEN_Q5Dnew"] = pd.Categorical(
    data["PEN_Q5Dnew"],
    categories=[
        "Very desirable",
        "Somewhat desirable",
        "Not very desirable",
        "Not at all desirable",
        "Don't know",
    ],
    ordered=True,
)

# %%
summary.create_single_question_summary_frame(data, "PEN_Q5Dnew")

# %% [markdown]
# #### Are you be willing to share your postcode with Experian for these purposes?
# - Response type: Text strings
# - Number of NA: 0
# - Cleaning:
#     - Convert to boolean

# %%
data["Q_consent"] = data["Q_consent"].map(
    {
        "Yes I am happy to share this information": True,
        "No, I do not wish to share this information": False,
    }
)

# %% [markdown]
# #### Thanks for agreeing to share this information.Please provide your Postcode in the box below.
# - Response type: Text strings
# - Number of NA: 0
# - Number of blanks (""): 1768 (those who did not give consent)
# - Cleaning:
#     - Convert to string type
#     - Convert blanks to "Not asked"
#     - Convert non-postcode responses to "Invalid response"
#     - Convert all postcode letter characters to uppercase
#     - Remove all spaces

# %%
data["Consent_1"] = data["Consent_1"].astype(str)

# %%
# Handling blanks
data["Consent_1"] = data["Consent_1"].replace(to_replace="", value="Did not consent")

# %%
# Identifying and handling invalid responses
# Include partial postcodes
# Invalid criteria: Does not start with a letter OR Does not contain a number in the first four characters

invalid_postcode_indices = data[
    ~data["Consent_1"].str.match(r"^[A-Za-z]", na=False)  # does not start with letter
    | ~data["Consent_1"]
    .str[:4]
    .str.contains(r"\d", na=False)  # does not contain number in first 4 chars
].index

# Remove "Did not consent"
not_asked_indices = data[data["Consent_1"] == "Did not consent"].index
invalid_postcode_indices = [
    item for item in invalid_postcode_indices if item not in not_asked_indices
]

# Create a new recoded column
data["Consent_1_recoded"] = data["Consent_1"]

for i in invalid_postcode_indices:
    data.loc[i, "Consent_1_recoded"] = "Invalid response"

# %%
# Verify all are invalid postcodes
for i in invalid_postcode_indices:
    print(data["Consent_1"][i])

# %%
# For valid postcodes: Remove space and change to all uppercase
data["Consent_1_recoded"] = data["Consent_1_recoded"].apply(
    lambda x: (
        x.replace(" ", "").upper()
        if x not in ["Did not consent", "Invalid response"]
        else x
    )
)

# %% [markdown]
# #### Are you...? (gender)
# - Response type: Text strings
# - Number of NA: 0
# - Cleaning:
#     - Convert to categorical

# %%
data["profile_gender"] = pd.Categorical(
    data["profile_gender"],
    categories=["Male", "Female"],
    ordered=False,
)

# %%
summary.create_single_question_summary_frame(data, "profile_gender")

# %% [markdown]
# #### Age
# - Response type: Text strings
# - Number of NA: 0
# - Cleaning:
#     - Convert to categorical

# %%
data["profile_julesage"] = pd.Categorical(
    data["profile_julesage"],
    categories=["18-24", "25-34", "35-44", "45-54", "55+"],
    ordered=True,
)

# %%
summary.create_single_question_summary_frame(data, "profile_julesage")

# %% [markdown]
# #### Social grade
# - Response type: Text strings
# - Number of NA: 0
# - Cleaning:
#     - Convert to categorical

# %%
data["profile_socialgrade_cie_rc"] = pd.Categorical(
    data["profile_socialgrade_cie_rc"],
    categories=["ABC1", "C2DE"],
    ordered=True,
)

# %%
summary.create_single_question_summary_frame(data, "profile_socialgrade_cie_rc")

# %% [markdown]
# #### Region
# - Response type: Text strings
# - Number of NA: 0
# - Cleaning:
#     - Convert to boolean
#
# - _1 North
# - _2 Midlands
# - _3 East
# - _4 London
# - _5 South
# - _6 England (NET)
# - _7 Wales
# - _8 Scotland
# - _9 Northern Ireland

# %%
gornewuk_options = [
    option for option in code_question_lookup.keys() if "gornewUK_" in option
]

# Modify full question labels in lookup
for option in gornewuk_options:
    code_question_lookup[option] = "Region: " + code_question_lookup.get(option)

# %%
# Convert to boolean
for option in gornewuk_options:
    data[option] = data[option].map(
        {
            "Yes": True,
            "No": False,
        }
    )

# %% [markdown]
# #### Working status
# - Response type: Text strings
# - Number of NA: 0
# - Cleaning:
#     - Convert to boolean
#
# - _1 Working full time
# - _2 Working part time
# - _3 ALL WORKERS (NET)
# - _4 Full time student
# - _5 Retired
# - _6 Unemployed
# - _7 Not working/ Other

# %%
work_stat_options = [
    option for option in code_question_lookup.keys() if "profile_work_stat_r_" in option
]

# Modify full question labels in lookup
for option in work_stat_options:
    code_question_lookup[option] = "Working Status: " + code_question_lookup.get(option)

# %%
# Convert to boolean
for option in work_stat_options:
    data[option] = data[option].map(
        {
            "Yes": True,
            "No": False,
        }
    )

# %% [markdown]
# #### Marital status
# - Response type: Text strings
# - Number of NA: 36 (Cannot identify whether they whether these were not asked based on a condition, or whether they were asked and did not give any answer)
# - Cleaning:
#     - Keep NaN as empty
#     - Convert to categorical

# %%
data["profile_marital_stat_r"] = pd.Categorical(
    data["profile_marital_stat_r"],
    categories=[
        "Married/ Civil Partnership",
        "Living as married",
        "Separated/ Divorced",
        "Widowed",
        "Never Married",
    ],
    ordered=False,
)

# %%
summary.create_single_question_summary_frame(data, "profile_marital_stat_r")

# %% [markdown]
# #### Children in Household
# - Response type: Text strings
# - Number of NA: 0
# - Cleaning:
#     - Convert to boolean
#
# - _1 0
# - _2 1
# - _3 2
# - _4 3+
# - _5 ALL WITH CHILDREN IN HOUSEHOLD (NET)
# - _6 Refused

# %%
children_options = [
    option
    for option in code_question_lookup.keys()
    if "profile_household_children_r_" in option
]

# Modify full question labels in lookup
for option in children_options:
    code_question_lookup[option] = "Children in Household: " + code_question_lookup.get(
        option
    )

# %%
# Convert to boolean
for option in children_options:
    data[option] = data[option].map(
        {
            "Yes": True,
            "No": False,
        }
    )

# %% [markdown]
# #### Parent/Guardian
# - Response type: Text strings
# - Number of NA: 0
# - Cleaning
#     - Convert to boolean
#
# - _1 Parent/ guardian (any age)
# - _2 Not parent/ guardian
# - _3 4 years and under
# - _4 5 to 11 years
# - _5 12 to 16 years
# - _6 17 to 18 years
# - _7 18 years and under
# - _8 Over 18 years

# %%
parent_options = [
    option for option in code_question_lookup.keys() if "omnibus_parents_rc3_" in option
]

# Modify full question labels in lookup
for option in parent_options:
    code_question_lookup[option] = "Parent/Guardian: " + code_question_lookup.get(
        option
    )

# %%
# Convert to boolean
for option in parent_options:
    data[option] = data[option].map({"Yes": True, "No": False})

# %%
# Collapse options to a single column
data = cleaning.collapse_select_all(
    data, parent_options, "omnibus_parents_rc3_collapsed", code_question_lookup
)

# Update question lookup
code_question_lookup["omnibus_parents_rc3_collapsed"] = "Parent/Guardian"

# %% [markdown]
# #### Social Media/ Messaging service (within the last month)
# - Response type: Text strings
# - Number of NA: 434 (Cannot identify whether they whether these were not asked based on a condition, or whether they were asked and did not give any answer)
# - Cleaning:
#     - Keep NaN as empty
#     - Convert "Yes"/"No" to "True"/"False" (string)
#
# - _1 Facebook
# - _2 X
# - _3 LinkedIn
# - _4 Pinterest
# - _5 Instagram
# - _6 Snapchat
# - _7 TikTok
# - _8 Facebook Messenger
# - _9 WhatsApp
# - _10 Skype

# %%
social_media_options = [
    option
    for option in code_question_lookup.keys()
    if "social_media_messaging_rc_" in option
]

# Modify full question labels in lookup
for option in social_media_options:
    code_question_lookup[option] = (
        "Social Media/ Messaging service (within the last month): "
        + code_question_lookup.get(option)
    )

# %%
# Convert to boolean
for option in social_media_options:
    data[option] = data[option].map(
        {
            "Yes": "True",
            "No": "False",
        }
    )

# %%
# Collapse options to a single column
data = cleaning.collapse_select_all(
    data,
    social_media_options,
    "social_media_messaging_rc_collapsed",
    code_question_lookup,
)

# Update question lookup
code_question_lookup["social_media_messaging_rc_collapsed"] = (
    "Social Media/ Messaging service (within the last month)"
)

# %% [markdown]
# #### Local authority / District / Unitary Authority
# - Response type: Text strings
# - Number of NA: 0
# - Number of unique values: 360
# - Cleaning:
#     - Convert to categorical

# %%
data["profile_oslaua"] = pd.Categorical(
    data["profile_oslaua"], categories=data["profile_oslaua"].unique()
)  # No set order

# %% [markdown]
# #### [Keep criteria]
# Note: These are used by YouGov to determine which responses to include.
#
# - Response type: Text strings
# - Number of NA: 0
# - Cleaning:
#     - Convert to boolean
#
# - _0 keep (All below must be True)
# - _1 profilesocialgradeciewtrc not empty
# - _2 profileeducationlevelrecode not empty
# - _3 profileworkstatr not empty
# - _4 profilehouseholdchildrenr not empty
# - _5 profileworktype not empty
# - _6 omnibusparentsrc3 not empty
# - _7 age >= 18 & profilejulesage not empty
# - _8 age >= 18 & profilebpcagesex not empty
# - _9 age >= 18

# %%
keep_options = [
    option for option in code_question_lookup.keys() if "UK18_Sept_2020_f_" in option
]

# Modify full question labels in lookup
for option in keep_options:
    code_question_lookup[option] = "Keep criteria: " + code_question_lookup.get(option)

# %%
# Convert to boolean
for option in keep_options:
    data[option] = data[option].map(
        {
            "Yes": True,
            "No": False,
        }
    )

# %% [markdown]
# ### Save cleaned data to parquet file

# %%
# Check final info
data.info()

# %%
cleaned_data_path = f"""/mnt/g/Shared drives/A Sustainable Future/1. Reducing household emissions/2. Projects Research Work/51.Identifying the next 3 million heat pump owners/YouGov survey data/Analysis ready data/{datetime.now().strftime("%Y%m%d")}_yougov_survey_clean_data.parquet"""

# %%
data.to_parquet(cleaned_data_path)

# %% [markdown]
# ### Save question code lookup to config

# %%
with open(
    f"{PROJECT_DIR}/asf_next_three_million_heat_pump_owners/config/base.yaml", "r"
) as yamlfile:
    existing_data = yaml.safe_load(yamlfile)
    existing_data["code_question_lookup"] = code_question_lookup

with open(
    f"{PROJECT_DIR}/asf_next_three_million_heat_pump_owners/config/base.yaml", "w"
) as yamlfile:
    yaml.dump(existing_data, yamlfile, default_flow_style=False)
