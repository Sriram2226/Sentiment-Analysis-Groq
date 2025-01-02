import streamlit as st
import requests

# Streamlit app layout
st.title("Sentiment Analysis Tool")
st.write(
    "Upload a CSV or Excel file containing reviews, and get the average sentiment scores."
)

# File uploader
uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])

# Backend API URL
backend_url = "http://localhost:8000/read_reviews"

if uploaded_file:
    # Display file information
    st.write("File uploaded successfully.")
    
    # Submit the file to the backend
    if st.button("Analyze Sentiments"):
        try:
            # Prepare the file for the POST request
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post(backend_url, files=files)
            
            # Check if the response is successful
            if response.status_code == 200:
                data = response.json()
                st.write("Average Sentiment Scores:")
                st.json(data["data"])
            else:
                st.error(f"Error: {response.json()['detail']}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
