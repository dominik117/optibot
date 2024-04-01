import streamlit as st
import optibot as ob 
import time
import threading
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()
st.session_state['api_key'] = os.getenv('OPENAI_API_KEY')

def st_run_optibot(df, api_key, context):
    topics = ob.main.run_optibot(df, api_key, context)
    st.session_state['topics'] = topics
    

st.title('OptiBot Optimization')
data_file = st.file_uploader("Upload your data (CSV format)", type=["csv"])

if data_file is not None:
    df = ob.load_data.load_csv(data_file)


    if 'column_selection_done' not in st.session_state:
        st.session_state['column_selection_done'] = False
    if 'question_cols' not in st.session_state:
        st.session_state['question_cols'] = []
    if 'answer_cols' not in st.session_state:
        st.session_state['answer_cols'] = []

    column_options = ['Select...'] + list(df.columns)

    col_option = st.radio("Are your questions and answers in separate columns or a single column?",
                          ('Separate Columns', 'Single Column'))

    
    if not st.session_state['column_selection_done']:

        if col_option == 'Separate Columns':
            question_cols = st.multiselect('Select one or more columns for questions:', column_options, key='question_cols')
            answer_cols = st.multiselect('Select one or more columns for answers:', column_options, key='answer_cols')
            if st.button('Finish Selection'):
                if st.session_state['question_cols'] and st.session_state['answer_cols']:
                    st.session_state['result_df'] = ob.load_data.merge_multiple_qa_columns(df, st.session_state['question_cols'], st.session_state['answer_cols'])
                    st.session_state['column_selection_done'] = True
                    #st.dataframe(st.session_state['result_df'])
        else:
            unified_col = st.selectbox('Select the column with both questions and answers:', column_options, index=0, key='unified_col')
            if st.session_state['unified_col'] != 'Select...':
                st.session_state['result_df'] = ob.load_data.extract_unified_column(df, st.session_state['unified_col'])
                st.session_state['column_selection_done'] = True
                #st.dataframe(st.session_state['result_df'])

    if st.session_state.get('column_selection_done'):
       
        if 'context' not in st.session_state or not st.session_state['context']:     
            st.session_state['context'] = st.text_input("In 1 to 5 words, what is the context of the conversations?")

        if st.session_state['column_selection_done'] and st.session_state['context']:
            st.write("The chatbot conversations have been selected, initiating analysis.")
            current_time = datetime.now()
            formatted_time = current_time.strftime("%A, %B %d, %Y %H:%M")
            st.write("Analysis started at: ", formatted_time)
            st.write("The estimated time for this analysis is usually around 2 hours. Please be patient.")

            st_run_optibot(st.session_state['result_df'], st.session_state['api_key'], st.session_state['context'])

            topics = st.session_state['topics']

            st.text(f"The analysis took {(topics.execution_time/60):.1f} minutes and {int(topics.resource_usage)} MBs of memory.")

            st.header("Results")

            st.subheader("Optimal Number of Topics")
            st.pyplot(topics.show_coherence_plot())
            st.caption("The optimal number of topics is determined by the highest coherence score.")

            st.subheader("Topic Modeling Results")
            st.dataframe(topics.topics_df)
            st.caption("The table shows the top words for each topic and their labels.")

            st.subheader("Topic Distribution")
            st.dataframe(topics.corpus_topic_df.sample(20))
            st.caption("The table shows the distribution of topics and their labels in the corpus, with a sample of 20 random conversations.")






