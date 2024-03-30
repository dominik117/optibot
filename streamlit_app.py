import streamlit as st
import optibot as ob  # Replace 'optibot' with your actual module name
import time

def prograss_bar():
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(100):
        status_text.text(f'Processing... {i+1}%')
        progress_bar.progress(i + 1)
        time.sleep(0.3)

    status_text.text('Processing... Done!')
    progress_bar.empty()
    st.success('Task completed!')



st.title('OptiBot Optimization')

data_file = st.file_uploader("Upload your data (CSV format)", type=["csv"])

if data_file is not None:
    df = ob.load_csv(data_file)

    # Initialize or reset session state variables
    if 'column_selection_done' not in st.session_state:
        st.session_state['column_selection_done'] = False

    column_options = ['Select...'] + list(df.columns)
    col_option = st.radio("Are your questions and answers in separate columns or a single column?",
                          ('Separate Columns', 'Single Column'))

    if not st.session_state['column_selection_done']:
        if col_option == 'Separate Columns':
            question_col = st.selectbox('Select the column for questions:', column_options, index=0, key='question_col')
            answer_col = st.selectbox('Select the column for answers:', column_options, index=0, key='answer_col')
            if st.session_state['question_col'] != 'Select...' and st.session_state['answer_col'] != 'Select...':
                result_df = ob.merge_qa_columns(df, st.session_state['question_col'], st.session_state['answer_col'])
                st.session_state['column_selection_done'] = True
                # st.dataframe(result_df)
        else:
            unified_col = st.selectbox('Select the column with both questions and answers:', column_options, index=0, key='unified_col')
            if st.session_state['unified_col'] != 'Select...':
                result_df = ob.extract_unified_column(df, st.session_state['unified_col'])
                st.session_state['column_selection_done'] = True
                # st.dataframe(result_df)

    # If the columns have been selected, you can add additional logic or display further content
    if st.session_state['column_selection_done']:
        # Further logic or display can go here
        # st.write("Column selection is complete.")
        prograss_bar()







# import streamlit as st
# import optibot as ob  # Replace 'optibot' with your actual module name

# st.title('OptiBot Optimization')

# data_file = st.file_uploader("Upload your data (CSV format)", type=["csv"])

# if data_file is not None:
#     df = ob.load_csv(data_file)

#     # Initialize session state variables if they don't exist
#     if 'question_col' not in st.session_state:
#         st.session_state['question_col'] = 'Select...'
#     if 'answer_col' not in st.session_state:
#         st.session_state['answer_col'] = 'Select...'
#     if 'unified_col' not in st.session_state:
#         st.session_state['unified_col'] = 'Select...'
    
#     column_options = ['Select...'] + list(df.columns)
#     col_option = st.radio("Are your questions and answers in separate columns or a single column?",
#                           ('Separate Columns', 'Single Column'))

#     if col_option == 'Separate Columns':
#         question_col = st.selectbox('Select the column for questions:', column_options, index=0, key='question_col')
#         answer_col = st.selectbox('Select the column for answers:', column_options, index=0, key='answer_col')
#         if st.session_state['question_col'] != 'Select...' and st.session_state['answer_col'] != 'Select...':
#             result_df = ob.merge_qa_columns(df, st.session_state['question_col'], st.session_state['answer_col'])
#             st.dataframe(result_df)
#     else:
#         unified_col = st.selectbox('Select the column with both questions and answers:', column_options, index=0, key='unified_col')
#         if st.session_state['unified_col'] != 'Select...':
#             result_df = ob.extract_unified_column(df, st.session_state['unified_col'])
#             st.dataframe(result_df)
#             # hide unified_col





# import streamlit as st
# import optibot as ob
# import pandas as pd


# st.title('OptiBot Optimization')

# data_file = st.file_uploader("Upload your data (CSV format)", type=["csv"])

# if data_file is not None:
#     df = ob.load_csv(data_file)
    
#     column_options = ['Select...'] + list(df.columns)

#     col_option = st.radio("Are your questions and answers in separate columns or a single column?",
#                           ('Separate Columns', 'Single Column'))

#     if col_option == 'Separate Columns':
#         question_col = st.selectbox('Select the column for questions:', column_options, index=0)
#         answer_col = st.selectbox('Select the column for answers:', column_options, index=0)
#         # Check if both columns are selected (not the placeholder)
#         if question_col != 'Select...' and answer_col != 'Select...':
#             result_df = ob.merge_qa_columns(df, question_col, answer_col)
#             # Stop showing the select
#     else:
#         unified_col = st.selectbox('Select the column with both questions and answers:', column_options, index=0)
#         # Check if the column is selected (not the placeholder)
#         if unified_col != 'Select...':
#             result_df = ob.extract_unified_column(df, unified_col)
#             st.text("BAM!")
#             unified_col = None

