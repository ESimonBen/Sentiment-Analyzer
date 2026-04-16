# Sentiment Analyzer

A simple NLP project that classifies text sentiment as **Positive**, **Negative**, or **Neutral** using a pretrained transformer model from Hugging Face.

---

## Overview

This project demonstrates practical usage of Natural Language Processing (NLP) by:
- Performing sentiment analysis on input text
- Handling invalid inputs (non-string and empty input)
- Evaluating model performance on a labeled test dataset
- Analyzing incorrect or uncertain predictions

---

## Model Used

- **Model:** `cardiffnlp/twitter-roberta-base-sentiment`
- **Library:** Hugging Face Transformers

### Why this model?
The model supports **three sentiment classes (Positive, Neutral, Negative)**, which avoids forcing neutral text into binary classifications and improves handling of ambiguous language.

---

## How It Works

1. Input text is validated:
   - Must be a string
   - Cannot be empty

2. The model predicts:
   - Sentiment label
   - Confidence score

3. A post-processing step:
   - Maps model labels to readable output
   - Downgrades low-confidence predictions to **Neutral**

---

## Project Structure

```
sentiment-analyzer/
│── app.py              # Core sentiment analysis logic
│── test_sentences.py   # Test dataset and evaluation script
│── requirements.txt
│── README.md
```

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

### Run a single test:

```bash
python app.py
```

### Run full evaluation:

```bash
python test_sentences.py
```

## 🧪 Test Results

The model was evaluated on a dataset consisting of:

- 4 Positive sentences  
- 4 Negative sentences  
- 4 Neutral sentences  
- Additional edge cases (negation, sarcasm, ambiguity)

### Example Output

```text
Text: I love this product!
Expected: Positive
Predicted: Positive (0.99)
Result: Correct
------------------------------------------------------------
```

### Final Accuracy

```text
76.47% (13 / 17)
```

---

## Error Analysis

### 1. "Great... just great." (Sarcasm)

- **Expected:** Negative  
- **Predicted:** Positive  
- **Reason:** The model struggles with sarcasm and interprets the word "great" as positive sentiment, ignoring contextual sarcasm.

---

### 2. "I guess it's okay." (Ambiguity)

- **Expected:** Neutral  
- **Predicted:** Positive  
- **Reason:** The model interprets mild approval as positive, even though the sentence expresses uncertainty and weak sentiment.

---

## Key Takeaways

- Transformer-based sentiment models are strong at clear sentiment classification but struggle with:
  - Sarcasm and irony  
  - Subtle or low-intensity sentiment  
  - Context-dependent meaning  

- Using a 3-class model (Positive / Neutral / Negative) improves handling of ambiguous or neutral statements compared to binary sentiment models.

---

## Demo

(Add your 1–2 minute screen recording link here)

---

## Notes

- The model is loaded once globally to improve performance and avoid repeated initialization overhead.
- On Windows systems, Hugging Face may display a symlink warning during model caching; this does not affect functionality.