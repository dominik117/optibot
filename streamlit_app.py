import streamlit as st
import optibot as ob 
import time
import threading
from datetime import datetime, timezone
from dotenv import load_dotenv
import os
from zoneinfo import ZoneInfo

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

if api_key:
    st.session_state['api_key'] = api_key
    st.session_state['api_key_received'] = True
else:
    if 'api_key_received' not in st.session_state or not st.session_state['api_key_received']:
        user_input = st.text_input("Enter your OpenAI API key:")
        if st.button('Submit API Key'):
            if user_input:
                st.session_state['api_key'] = user_input
                st.session_state['api_key_received'] = True
                st.rerun()
            else:
                st.warning('Please enter a valid API key.')
    else:
        st.success("Thank you, key received.")

@st.cache(allow_output_mutation=True)
def st_run_optibot(df, api_key, context):
    if 'analysis_started' not in st.session_state:
        st.session_state.analysis_started = True
        print("Starting analysis function from streamlit...")
        topics = ob.main.run_optibot(df, api_key, context)
        st.session_state['topics'] = topics
        st.session_state['analysis_done'] = True
        st.session_state.analysis_started = False

if st.button('Reset App'):
    st.session_state.clear()

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

        if st.session_state['column_selection_done'] and st.session_state['context'] and not st.session_state.get('analysis_done', False):
            st.write("The chatbot conversations have been selected, initiating analysis.")

            current_time = datetime.now(tz=timezone.utc)
            zurich_time = current_time.astimezone(ZoneInfo("Europe/Zurich"))
            formatted_time = zurich_time.strftime("%A, %B %d, %Y %H:%M %Z")
            st.write("Analysis started at: ", formatted_time)
            st.write("The estimated time for this analysis is usually around 2 hours. Please be patient.")

            if 'analysis_done' not in st.session_state or not st.session_state.analysis_done:
                if data_file and 'result_df' in st.session_state and st.session_state['context'] and not st.session_state.get('analysis_started', False):
                    st_run_optibot(st.session_state['result_df'], st.session_state['api_key'], st.session_state['context'])

            # st_run_optibot(st.session_state['result_df'], st.session_state['api_key'], st.session_state['context'])

            topics = st.session_state['topics']

            st.text(f"The analysis took {(topics.execution_time/60):.1f} minutes and {int(topics.resource_usage)} MBs of memory.")

            st.header("Results")

            st.subheader("Optimal Number of Topics")
            st.pyplot(topics.show_coherence_plot())
            st.caption("The optimal number of topics is determined by the highest coherence score.")

            st.subheader("Topic Modeling Results")
            st.dataframe(topics.topics_df)
            st.caption("The table shows the top words for each topic and their labels.")

            st.subheader("Conversation Assessment")
            st.dataframe(topics._assessed_conversations_df.sample(20))
            st.caption("The table shows a sample of the final topic labeling and assessment of the conversations.")

            st.subheader("Topic Distribution")
            st.plotly_chart(topics.show_topic_distribution_plot())
            st.caption("The plot shows the distribution of topics across the conversations.")

            st.subheader("Final Insights")
            self_final_insights = topics.final_insights
            for topic, content in self_final_insights.items():
                st.markdown(f"<h4>Topic: {topic}</h4>", unsafe_allow_html=True)
                st.write(f"Insight: {content['summary']}")  
                st.pyplot(content['figure'])
                st.caption("The figure shows the average scores for each criterion in the assessment.")
                break_line = st.empty()
                break_line.write("---")




