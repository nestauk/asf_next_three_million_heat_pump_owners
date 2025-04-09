import pandas as pd


def collapse_select_all(
    data: pd.DataFrame,
    select_all_columns: list[str],
    collapsed_column_name: str,
    question_lookup: dict,
) -> pd.DataFrame:
    """
    Collapses multiple boolean 'Select all that apply' columns into a single column
    containing a list of selected option labels based on a lookup dictionary.

    For each row, this function checks the specified columns for true values
    (`True` or `"True"`), and maps the column names to labels using the `question_lookup`
    dictionary. The resulting list of selected options is stored in a new column.
    """
    data[collapsed_column_name] = data[select_all_columns].apply(
        lambda row: [
            question_lookup[col].split(": ")[1]
            for col in select_all_columns
            if row[col] == True or row[col] == "True"
        ],
        axis=1,
    )
    return data
