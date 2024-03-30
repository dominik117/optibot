import streamlit as st
import optibot as ob 
import time
import threading

def st_run_optibot(df):
    topics = ob.main.run_optibot(df)
    return topics 

def start_analysis(result_df):
    if 'analysis_started' not in st.session_state or not st.session_state['analysis_started']:
        st.session_state['analysis_started'] = True
        st.session_state['start_time'] = time.time()
        st.session_state['analysis_completed'] = False
        
        # Start the analysis in a separate thread
        def run_analysis():
            topics = st_run_optibot(result_df)
            st.session_state['topics'] = topics  # Store the result in the session state
            st.session_state['analysis_completed'] = True
        
        analysis_thread = threading.Thread(target=run_analysis)
        analysis_thread.start()




st.title('OptiBot Optimization')
data_file = st.file_uploader("Upload your data (CSV format)", type=["csv"])

if data_file is not None:
    df = ob.load_data.load_csv(data_file)

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
                result_df = ob.load_data.merge_qa_columns(df, st.session_state['question_col'], st.session_state['answer_col'])
                st.session_state['column_selection_done'] = True
                #st.dataframe(result_df)
        else:
            unified_col = st.selectbox('Select the column with both questions and answers:', column_options, index=0, key='unified_col')
            if st.session_state['unified_col'] != 'Select...':
                result_df = ob.load_data.extract_unified_column(df, st.session_state['unified_col'])
                st.session_state['column_selection_done'] = True
                #st.dataframe(result_df)


    if st.session_state['column_selection_done']:
        st.write("The chatbot conversations have been selected, initiating analysis.")
        time.sleep(5)

        start_analysis(result_df)

        if 'analysis_started' in st.session_state and st.session_state['analysis_started']:
            time_placeholder = st.empty()
            estimated_time_placeholder = st.empty()

            while not st.session_state.get('analysis_completed', False):
                elapsed_time = int(time.time() - st.session_state['start_time'])
                minutes = elapsed_time // 60
                seconds = elapsed_time % 60
                time_placeholder.markdown(f"**Elapsed Time:** {minutes} minutes and {seconds} seconds")
                estimated_time_placeholder.markdown("The estimated time for this analysis is usually around 15 minutes. Please be patient.")
                time.sleep(1)
            
            # Clear the placeholder once the analysis is complete
            estimated_time_placeholder.empty()


        topics = st.session_state['topics']

        st.text(f"The analysis took {(topics.execution_time/60):.1f} minutes and {int(topics.resource_usage)} MBs of memory.")

        st.header("Results")

        st.subheader("Optimal Number of Topics")
        st.pyplot(topics.coherence_plot)
        st.caption("The optimal number of topics is determined by the highest coherence score.")

        st.subheader("Topic Modeling Results")
        st.dataframe(topics.topics_df)
        st.caption("The table shows the top words for each topic.")

        st.subheader("Topic Distribution")
        st.dataframe(topics.corpus_topic_df.sample(20))
        st.caption("The table shows the distribution of topics in the corpus, with a sample of 20 random conversations.")





