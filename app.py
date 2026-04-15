# app.py

from transformers import pipeline

# Mapping from model output labels to human-readable sentiment
LABEL_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
}

def validate_input(text):
    """
    Validates the input text.

    Ensures the input is:
    - A string
    - Not empty or whitespace-only

    Raises:
        ValueError: If input is invalid
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    if text.strip() == "":
        raise ValueError("Input cannot be empty")


def get_sentiment(text):
    """
    Classifies the sentiment of the given text.

    Steps:
    1. Validate input
    2. Run model inference
    3. Map model label to readable sentiment
    4. Adjust for low-confidence predictions

    Args:
        text (str): Input text to analyze

    Returns:
        tuple: (sentiment (str), confidence score (float))
    """
    validate_input(text)

    # Run sentiment analysis model
    result = classifier(text)[0]

    # Extract label and confidence score safely
    label = result.get("label")
    score = result.get("score", 0)

    # Convert model label to readable sentiment
    sentiment = LABEL_MAP.get(label, "Neutral")

    # If confidence is low, downgrade to Neutral (unless already Neutral)
    if score <= 0.6 and sentiment != "Neutral":
        sentiment = "Neutral"

    return sentiment, score


def load_model():
    """
    Loads the Hugging Face sentiment analysis pipeline.

    Uses a 3-class sentiment model (Positive, Neutral, Negative)
    to better handle ambiguous or neutral text.

    Returns:
        Hugging Face pipeline object
    """
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )


# Load model once globally to avoid repeated initialization (performance optimization)
classifier = load_model()