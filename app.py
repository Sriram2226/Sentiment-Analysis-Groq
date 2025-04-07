from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import pandas as pd
import os
import json
import re
from io import BytesIO
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "How to use": "POST a CSV or EXCEL file with a column named 'Review' to /read_reviews. "
                      "The API returns average POSITIVE, NEGATIVE, and NEUTRAL sentiment scores."
    }

@app.post("/read_reviews")
def read_reviews(file: UploadFile):
    # Log file info
    print(f"Received file: {file.filename}")

    try:
        contents = file.file.read()
        print(f"File size: {len(contents)} bytes")
    except Exception as e:
        print("Error reading file:", e)
        raise HTTPException(status_code=400, detail="Failed to read uploaded file.")

    # Try loading file into DataFrame
    try:
        if file.filename.endswith(".xlsx"):
            df = pd.read_excel(BytesIO(contents))
        elif file.filename.endswith(".csv"):
            df = pd.read_csv(BytesIO(contents))
        else:
            raise ValueError("Unsupported file format.")
    except Exception as e:
        print("File loading error:", e)
        raise HTTPException(status_code=400, detail="Error reading file content or incorrect format.")

    # Validate 'Review' column
    if "Review" not in df.columns:
        raise HTTPException(status_code=400, detail="Missing 'Review' column in the file.")

    # Extract reviews and format for model
    reviews = df["Review"].dropna().astype(str).tolist()
    if not reviews:
        raise HTTPException(status_code=400, detail="No valid reviews found in the file.")

    formatted_reviews = ', '.join(f"{i}: '{r}'" for i, r in enumerate(reviews))
    print(f"Formatted Reviews:\n{formatted_reviews[:500]}...")  # Print a snippet for debug

    # Initialize Groq client
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Query model
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a DATA ANALYST capable of sentiment analysis from a list of reviews. "
                        "Return only JSON with this format: "
                        "{\"0\": {\"POSITIVE\": float, \"NEGATIVE\": float, \"NEUTRAL\": float}, ...}"
                    )
                },
                {
                    "role": "user",
                    "content": formatted_reviews
                }
            ]
        )

        raw_response = response.choices[0].message.content
        print("Raw model response:\n", raw_response)

        # Extract JSON safely
        json_str = re.search(r'\{.*\}', raw_response, re.DOTALL).group()
        review_scores = json.loads(json_str)

    except Exception as e:
        print("Model response parsing error:", e)
        raise HTTPException(status_code=400, detail="Invalid response from sentiment model. Try reuploading.")

    # Calculate average sentiment
    try:
        total = len(review_scores)
        pos_sum = sum(item["POSITIVE"] for item in review_scores.values())
        neg_sum = sum(item["NEGATIVE"] for item in review_scores.values())
        neu_sum = sum(item["NEUTRAL"] for item in review_scores.values())

        analysis = {
            "positive": round(pos_sum / total, 4),
            "negative": round(neg_sum / total, 4),
            "neutral": round(neu_sum / total, 4)
        }

        return {"data": analysis}

    except Exception as e:
        print("Error computing sentiment averages:", e)
        raise HTTPException(status_code=500, detail="Failed to compute sentiment analysis.")
