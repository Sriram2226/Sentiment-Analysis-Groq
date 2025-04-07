const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const resultsDiv = document.getElementById("results");
const errorDiv = document.getElementById("error");

const positive = document.getElementById("positive");
const negative = document.getElementById("negative");
const neutral = document.getElementById("neutral");

// Replace this with your actual backend URL
const API_URL = "https://sentiment-analysis-groq.onrender.com/read_reviews";
// const API_URL = "http://localhost:8000/read_reviews";

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    resultsDiv.classList.add("hidden");
    errorDiv.classList.add("hidden");

    const file = fileInput.files[0];
    if (!file) {
        errorDiv.textContent = "Please select a file.";
        errorDiv.classList.remove("hidden");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Failed to analyze the file. Please ensure it contains a 'Review' column.");
        }

        const result = await response.json();
        positive.textContent = (result.data.positive * 100).toFixed(2) + "%";
        negative.textContent = (result.data.negative * 100).toFixed(2) + "%";
        neutral.textContent = (result.data.neutral * 100).toFixed(2) + "%";

        resultsDiv.classList.remove("hidden");
    } catch (error) {
        errorDiv.textContent = error.message;
        errorDiv.classList.remove("hidden");
    }
});
