import streamlit as st
from summarizer import article_summarize, word_cloud  # Import your functions from summarizer.py

st.title("Text Summarizer Web App")
st.info("Please note that while giving input, the number lines should be more than 7")
# Add Streamlit text input and summarization button here
user_input_text = st.text_area("Enter the text you want to summarize:", height=300)

if st.button("Summarize") and user_input_text:
    # Split the input text into lines and count the number of lines
    lines = user_input_text.splitlines()
    num_lines = len(lines)

    # Check if the input contains at least 7 lines
    if num_lines < 7:
        st.warning("Please enter a longer text with at least 7 lines for better summarization.")
    else:
        # Call your summarization function here
        summarized_text = article_summarize(user_input_text)

        # Display the summarized text
        st.subheader("Summary:")
        st.write(summarized_text[0])  # Display the first summary

