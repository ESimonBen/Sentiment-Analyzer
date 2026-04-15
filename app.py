#app.py

from transformers import pipeline

# Label Map
LABEL_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
}

# Validates input (Checks for things like non-strings or empty strings)
def validate_input(text):
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    if text.strip() == "":
        raise ValueError("Input cannot be empty")

# Sentiment retrieval function
def get_sentiment(text):
    validate_input(text)

    result = classifier(text)[0]

    label = result.get("label")
    score  = result.get("score")

    sentiment = LABEL_MAP.get(label, "Neutral")

    # Checks the label of the model to determine the sentiment
    # (LABEL_0 = Negative, LABEL_1 = Neutral, LABEL_2 = Positive)
    if score <= .6:
        sentiment = "Neutral"

    return sentiment, score

def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )

# Model is loaded once globally to prevent constant calls
classifier = load_model()

if __name__ == "__main__":
    test_sentences = [
        "I love this!",
        "This is terrible.",
        "People can be... weird and interesting sometimes.",
        "The meeting is at 3 PM."
    ]

    for sentence in test_sentences:
        sentiment, score = get_sentiment(sentence)
        print(f"{sentence} -> {sentiment} ({score:.2f})")