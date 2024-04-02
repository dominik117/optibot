import pandas as pd
from . import load_data
from .optibot_modeling import OptiBotModeling
import time

def run_optibot(df, api_key, context):
    topics = OptiBotModeling(df, api_key, context, start_topic_count=5, end_topic_count=12)
    topics.fit()
    print("Analysis completed")
    return topics

def main():
    df = load_data.upload_data()
    if df is not None:
        selected_df = load_data.select_columns(df)

def main_streamlit(df=None):
    return None

if __name__ == "__main__":
    main()
