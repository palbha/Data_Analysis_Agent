import streamlit as st
import pandas as pd
from agent import get_agent, run_query

st.set_page_config(page_title="Data Analysis Agent", layout="wide")

st.title("📊 Open Source Data Analysis Agent")
st.markdown("Upload a CSV file and ask questions about your data. The agent will write and execute Python code to answer your queries while tracing its thoughts to Langfuse.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    st.info("Make sure your API keys (HuggingFace and Langfuse) are set in the `.env` file before proceeding.")
    
# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head())
        
        # Initialize the agent in session state to persist it across queries
        if 'agent' not in st.session_state:
            with st.spinner("Initializing Agent..."):
                try:
                    agent, langfuse_handler = get_agent(df)
                    st.session_state.agent = agent
                    st.session_state.langfuse_handler = langfuse_handler
                    st.success("Agent initialized successfully!")
                except Exception as e:
                    st.error(f"Error initializing agent: {e}")
                    st.stop()
                    
        # Chat interface
        st.write("### Ask Questions")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("E.g., What is the average value in column X?"):
            # Display user message in chat message container
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Analyzing..."):
                    try:
                        # Run query through the agent
                        response = run_query(
                            st.session_state.agent, 
                            st.session_state.langfuse_handler, 
                            prompt
                        )
                        message_placeholder.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
