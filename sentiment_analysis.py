from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dataclasses import dataclass
import json
from typing import Tuple

@dataclass
class Mood:
    textblob_classification: str
    vader_classification: str
    textblob_sentiment: float
    vader_sentiment: float

THRESHOLD_TEXTBLOB = 0.33
THRESHOLD_VADER = 0.1

def get_sentiment_textblob(text: str, threshold: float = THRESHOLD_TEXTBLOB) -> Tuple[str, float]:
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment >= threshold:
        classification = "Positive"
    elif sentiment <= -threshold:
        classification = "Negative"
    else:
        classification = "Neutral"
    return classification, sentiment

def get_sentiment_vader(text: str, threshold: float = THRESHOLD_VADER) -> Tuple[str, float]:
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    if compound_score >= threshold:
        classification = "Positive"
    elif compound_score <= -threshold:
        classification = "Negative"
    else:
        classification = "Neutral"
    return classification, compound_score

def load_dataset_from_json(file_path: str) -> list:
    try:
        with open(file_path, "r") as json_file:
            dataset = json.load(json_file)
        return dataset
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON - {file_path}")
        return []

def analyze_text():
    while True:
        text = input("Enter text for sentiment analysis (or 'quit' to exit): ").strip()
        if text.lower() == "quit":
            break
        textblob_classification, textblob_sentiment = get_sentiment_textblob(text)
        vader_classification, vader_sentiment = get_sentiment_vader(text)
        mood = Mood(textblob_classification, vader_classification, textblob_sentiment, vader_sentiment)
        print(f"TextBlob - Classification: {mood.textblob_classification}, Sentiment: {mood.textblob_sentiment}")
        print(f"VADER - Classification: {mood.vader_classification}, Sentiment: {mood.vader_sentiment}")

def analyze_dataset(dataset: list):
    if not dataset:
        print("Dataset is empty or invalid.")
        return
    textblob_correct = 0
    vader_correct = 0
    total_samples = len(dataset)
    for sample in dataset:
        text = sample.get("text", "")
        ground_truth = sample.get("sentiment", "")
        textblob_classification, _ = get_sentiment_textblob(text)
        vader_classification, _ = get_sentiment_vader(text)
        if textblob_classification == ground_truth:
            textblob_correct += 1
        if vader_classification == ground_truth:
            vader_correct += 1
    textblob_accuracy = (textblob_correct / total_samples) * 100
    vader_accuracy = (vader_correct / total_samples) * 100
    print(f"TextBlob Accuracy: {textblob_accuracy:.2f}%")
    print(f"VADER Accuracy: {vader_accuracy:.2f}%")

def main():
    print("Choose an option:")
    print("1. Analyze text input")
    print("2. Analyze the dataset")
    choice = input("Enter your choice (1 or 2): ").strip()
    if choice == "1":
        analyze_text()
    elif choice == "2":
        dataset_path = input("Enter the dataset file path: ").strip()
        dataset = load_dataset_from_json(dataset_path)
        analyze_dataset(dataset)
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == '__main__':
    main()
