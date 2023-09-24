import streamlit as st
from summarizer import article_summarize, word_cloud  # Import your functions from summarizer.py

st.title("Text Summarizer")

# Add Streamlit text input and summarization button here
user_input_text = st.text_area("Enter the text you want to summarize:", height=300)

if st.button("Summarize") and user_input_text:
    # Call your summarization function here
    summarized_text = article_summarize(user_input_text)

    # Display the summarized text
    st.subheader("Summary:")
    st.write(summarized_text[0])  # Display the first summary

