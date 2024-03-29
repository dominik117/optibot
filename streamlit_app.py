import streamlit as st
import optibot as ob
import pandas as pd

st.title('OptiBot Optimization')

# Main button
if st.button('Start Optimization with OptiBot'):

    data_file = st.file_uploader("Upload your data (CSV format)", type=["csv"])
    if data_file is not None:
        
        df = ob.load_csv(data_file)


        # Option to select one or two columns
        col_option = st.radio("Are your questions and answers in separate columns or a single column?",
                              ('Separate Columns', 'Single Column'))

        if col_option == 'Separate Columns':
            question_col = st.selectbox('Select the column for questions:', df.columns)
            answer_col = st.selectbox('Select the column for answers:', df.columns)
            result_df = ob.extract_original_columns(df, question_col, answer_col)

        else:
            unified_col = st.selectbox('Select the column with both questions and answers:', df.columns)
            result_df = ob.extract_original_columns(df, unified_col)



        # Display the DataFrame
        st.dataframe(result_df)
