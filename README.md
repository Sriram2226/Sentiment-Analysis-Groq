Here is the documentation for your project:

---

# **Sentiment Analysis Tool with Groq API**

## **Project Overview**

This project is a sentiment analysis tool that processes customer reviews uploaded in CSV or Excel format and returns the average sentiment scores (Positive, Negative, Neutral) using the Groq API. It is composed of a FastAPI backend to handle file uploads and sentiment analysis, a frontend built with Streamlit for user interaction, and uses Groq's AI model to perform sentiment analysis on the reviews.

## **Directory Structure**

```
Sriram2226-Sentiment-Analysis-Groq/
    ├── app.py                # FastAPI backend for handling requests
    ├── commands.txt          # Commands to run the project
    ├── customer_reviews.xlsx # Example input file with reviews
    ├── frontend.py           # Streamlit frontend for uploading files and displaying results
    └── requirements.txt      # Required dependencies for the project
```

---

## **Files**

### **1. app.py** (Backend - FastAPI)

This file contains the FastAPI backend that processes the uploaded reviews and returns sentiment scores.

#### **Key Functions:**

- **`/` (GET request):**
  - Displays basic instructions on how to use the API.
  
- **`/read_reviews` (POST request):**
  - Accepts a file containing reviews.
  - Processes the reviews and returns the average sentiment scores (positive, negative, and neutral).
  - Requires the file to have a column named `Review`.

#### **Process:**
- Loads the file (CSV or Excel).
- Extracts reviews from the file.
- Sends the reviews to Groq's sentiment analysis model.
- Calculates the average sentiment scores from the response.
- Returns the calculated sentiment scores in JSON format.

---

### **2. commands.txt**

This file contains the commands to run the application and set up the environment.

#### **Commands:**

- **Activate the virtual environment:**
  ```
  env/Scripts/Activate
  ```
  
- **Run FastAPI application with Uvicorn:**
  ```
  uvicorn app:app --reload
  ```
  
- **Example of how to make a POST request to the API:**
  ```
  curl -X POST -F "file=@customer_reviews.xlsx" http://localhost:8000/read_reviews
  ```

---

### **3. frontend.py** (Frontend - Streamlit)

This file contains the frontend built using Streamlit. It allows users to upload a file containing reviews and view the sentiment analysis results.

#### **Key Features:**
- Title and description for the tool.
- File uploader to allow the user to upload CSV or Excel files.
- Button to trigger the sentiment analysis.
- Displays the average sentiment scores after analysis.

---

### **4. requirements.txt**

This file contains the list of required Python packages to run the project. Example content:

```
fastapi==0.95.0
groq==0.5.1
pandas==1.5.3
requests==2.28.2
streamlit==1.14.0
python-dotenv==0.21.0
uvicorn==0.18.2
```

---

## **Setup Instructions**

### **1. Install Dependencies**

To get started, clone the repository and install the required dependencies.

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Sriram2226-Sentiment-Analysis-Groq.git
   ```
   
2. Navigate to the project directory:
   ```
   cd Sriram2226-Sentiment-Analysis-Groq
   ```

3. Create a virtual environment:
   ```
   python -m venv env
   ```

4. Activate the virtual environment:
   - On Windows:
     ```
     env\Scripts\Activate
     ```
   - On Linux/Mac:
     ```
     source env/bin/activate
     ```

5. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

### **2. Run the Backend**

1. Run the FastAPI application with Uvicorn:
   ```
   uvicorn app:app --reload
   ```

   The application will be available at `http://localhost:8000`.

### **3. Run the Frontend**

1. Run the Streamlit app:
   ```
   streamlit run frontend.py
   ```

   The frontend will be available at `http://localhost:8501`.

---

## **Usage Instructions**

1. **Upload Reviews File**:
   - The file should be in CSV or Excel format and must contain a column named `Review`.
   
2. **Analyze Sentiments**:
   - After uploading the file, click the "Analyze Sentiments" button.
   - The tool will process the reviews and display the average sentiment scores (positive, negative, and neutral).

---

## **API Details**

### **GET `/`**
Returns the usage instructions for the API.

**Response:**
```json
{
  "How to use": "API takes in a CSV or EXCEL file containing reviews with a column named 'Review' and returns the average POSITIVE, NEGATIVE and NEUTRAL sentiment score of the reviews."
}
```

### **POST `/read_reviews`**
Takes a CSV or Excel file with reviews and returns sentiment analysis results.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Body**: Upload a file with a column `Review`.

**Response:**
```json
{
  "data": {
    "positive": 0.72,
    "negative": 0.18,
    "neutral": 0.10
  }
}
```

---

## **Environment Variables**

The project requires the `GROQ_API_KEY` to interact with the Groq API.

### **Steps to set environment variable:**

1. Create a `.env` file in the project root directory.
2. Add your Groq API key to the `.env` file:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

---

## **Error Handling**

- If the file format is incorrect, the backend will return a `400` status code with a message indicating the error.
- If the required column `Review` is not found, a `400` status code will be returned with a message indicating the missing column.

---

## **Contributing**

If you'd like to contribute to this project, feel free to fork the repository, make your changes, and submit a pull request.

---


Feel free to adjust or add any specific details as needed!
