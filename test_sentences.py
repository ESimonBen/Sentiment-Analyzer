# test_sentences.py

from app import get_sentiment

# Test dataset:
# Includes standard cases (positive, negative, neutral)
# and more challenging "curveball" cases to evaluate model robustness
test_data = [
    # Positive examples
    ("I love this product!", "Positive"),
    ("This is an amazing experience.", "Positive"),
    ("I am very happy with the results.", "Positive"),
    ("Everything worked perfectly!", "Positive"),

    # Negative examples
    ("I hate this so much.", "Negative"),
    ("This is the worst service ever.", "Negative"),
    ("I am very disappointed.", "Negative"),
    ("Nothing worked as expected.", "Negative"),

    # Neutral examples
    ("The sky is blue.", "Neutral"),
    ("I went to the store today.", "Neutral"),
    ("The meeting is at 3 PM.", "Neutral"),
    ("The document is on the table", "Neutral"),

    # Curveball / edge cases
    ("I am not unhappy.", "Neutral"),           # Double negation
    ("That's actually not bad.", "Positive"),   # Double negation (positive leaning)
    ("This is fine.", "Neutral"),               # Subtle sentiment
    ("Great... just great.", "Negative"),       # Sarcasm
    ("I guess it's okay.", "Neutral")           # Ambiguous / low sentiment
]

# Stores incorrectly classified cases for later analysis
incorrect_cases = []


def run_tests():
    """
    Runs sentiment analysis on all test cases.

    Outputs:
    - Predicted vs expected sentiment
    - Confidence score
    - Per-case correctness
    - Overall accuracy
    - Summary of incorrect predictions
    """
    print("=" * 60)
    print("Sentiment Analysis Test Results")
    print("=" * 60)

    correct = 0

    for text, expected in test_data:
        predicted, score = get_sentiment(text)

        # Check if prediction matches expected label
        is_correct = predicted == expected

        if is_correct:
            correct += 1
        else:
            # Store incorrect predictions for later review
            incorrect_cases.append((text, expected, predicted, score))

        # Print detailed result for each test case
        print(f"Text: {text}")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted} ({score:.2f})")
        print(f"Result: {'Correct' if is_correct else 'Incorrect'}")
        print("-" * 60)

    # Calculate and display overall accuracy
    accuracy = correct / len(test_data) * 100
    print(f"Final Accuracy: {accuracy:.2f}% ({correct}/{len(test_data)})")

    # Display incorrect predictions for analysis
    print("\nIncorrect Predictions:")
    print("-" * 60)
    for text, expected, predicted, score in incorrect_cases:
        print(f"Text: {text}")
        print(f"Expected: {expected}, Predicted: {predicted} ({score:.2f})")
        print("-" * 60)


if __name__ == "__main__":
    run_tests()