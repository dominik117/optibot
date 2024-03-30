import pandas as pd
from . import load_data
from . import topic_modeling


def run_optibot(df):
    topics = topic_modeling.OptiBotTopicModeling(df, start_topic_count=3, end_topic_count=12)
    topics.fit()
    return topics

def main():
    df = load_data.upload_data()
    if df is not None:
        selected_df = load_data.select_columns(df)

def main_streamlit(df=None):
    return None

if __name__ == "__main__":
    main()
