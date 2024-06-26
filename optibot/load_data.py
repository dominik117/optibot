import pandas as pd
# from tkinter import Tk, filedialog
from . import local_utils

def load_csv(data_file):
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(data_file)

def merge_multiple_qa_columns(df, question_cols, answer_cols):
    df['conversation'] = "Question: " + df[question_cols].apply(lambda x: ' '.join(x.astype(str)), axis=1) + " Answer: " + df[answer_cols].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    return df

def extract_unified_column(df, col):
    df["conversation"] = df[col]
    return df

def select_columns(df):
    """Ask the user to select columns for analysis."""
    print("\nColumns available in the dataset:")
    print(list(df.columns))

    col_option = local_utils.ask_radio_question(
        "Are your questions and answers in separate columns or a single column?",
        ["Separate Columns", "Single Column"]
    )

    print(f"You selected: {col_option}")

    if col_option == "Separate Columns":
        question_col = local_utils.ask_radio_question(
            "Select the column corresponding to the question text:",
            [list(df.columns)]
        )
        answer_col = local_utils.ask_radio_question(
            "Select the column corresponding to the ChatBot's answer text:",
            [list(df.columns)]
        )
        return merge_multiple_qa_columns(df, question_col, answer_col)

    if col_option == "Single Column":
        col = local_utils.ask_radio_question(
            "Select the column corresponding to the question and ChatBot's answer text:",
            [list(df.columns)]
        )
        return extract_unified_column(df, col)
