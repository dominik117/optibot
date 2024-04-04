import pandas as pd
from . import load_data
from .optibot_modeling import OptiBotModeling
import time

def run_optibot(df, api_key, context):
    print("Starting analysis...")
    topics = OptiBotModeling(df, api_key, context, start_topic_count=5, end_topic_count=18)
    topics.fit()
    print("Analysis completed")
    return topics

def main():
    print("Termnal support under development, for the Streamlit version, please run: poetry run optibot")
    print("Thank you for your understanding.")

def main_streamlit(df=None):
    return None

if __name__ == "__main__":
    main()
