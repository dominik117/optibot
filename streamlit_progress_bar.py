import streamlit as st
import threading
import time

# Function to simulate a long-running task
def long_running_task():
    # Simulate a long task
    time.sleep(10)
    # Notify the completion via a Streamlit command
    st.success('Task completed!')

# Function to display a loading message
def display_loading_message(placeholder):
    i = 0
    while True:
        # Update the placeholder with a loading message
        placeholder.text(f'Processing{"." * (i % 4)} (This may take a while)')
        time.sleep(0.3)
        i += 1

# Create a placeholder for the loading message
loading_placeholder = st.empty()

# Start a thread for the long-running task
task_thread = threading.Thread(target=long_running_task)
task_thread.start()

# Display the loading message while the task is running
display_loading_message(loading_placeholder)

# Wait for the thread to complete
task_thread.join()

# Clear the loading message
loading_placeholder.empty()
