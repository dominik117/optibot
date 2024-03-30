import pandas as pd
import load_data

def main():
    df = load_data.upload_data()
    if df is not None:
        selected_df = load_data.select_columns(df)

def main_streamlit(df=None):
    

if __name__ == "__main__":
    main()
