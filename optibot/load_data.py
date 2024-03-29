import pandas as pd

def load_csv(data_file):
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(data_file)


def extract_original_columns(df, question_col, answer_col=None):
    """
    Extract and return the specified columns from the DataFrame.
    If answer_col is None, it assumes questions and answers are in the same column.
    """
    if answer_col:
        return df[[question_col, answer_col]]
    return df[[question_col]]