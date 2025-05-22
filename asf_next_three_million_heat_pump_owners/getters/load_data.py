import pandas as pd
from asf_next_three_million_heat_pump_owners import config


def get_analysis_ready_data(
    uk_data_path: str = f"""/mnt/g/Shared drives/A Sustainable Future/1. Reducing household emissions/2. Projects Research Work/51.Identifying the next 3 million heat pump owners/YouGov survey data/Analysis ready data/20250409_yougov_survey_clean_data.parquet""",
    wales_data_path: str = f"""/mnt/g/Shared drives/A Sustainable Future/1. Reducing household emissions/2. Projects Research Work/51.Identifying the next 3 million heat pump owners/YouGov survey data/Analysis ready data/20250514_yougov_survey_clean_data_wales_boost.parquet""",
) -> tuple[pd.DataFrame, pd.DataFrame]:

    return pd.read_parquet(uk_data_path, engine="pyarrow"), pd.read_parquet(
        wales_data_path
    )
