from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dataclasses import dataclass

@dataclass
class Mood:
    textblob_classification: str
    vader_classification: str
    textblob_sentiment: float
    vader_sentiment: float

THRESHOLD_TEXTBLOB = 0.33
THRESHOLD_VADER = 0.1

def get_sentiment_textblob(text: str, threshold: float) -> Mood:
    sentiment = TextBlob(text).sentiment.polarity

    if sentiment >= threshold:
        classification = "Positive"
    elif sentiment <= -threshold:
        classification = "Negative"
    else:
        classification = "Neutral"

    return classification, sentiment

def get_sentiment_vader(text: str, threshold: float) -> Mood:
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

def main():
    print("Enter 'quit' to exit sentiment analysis")

    while True:
        text = input("Enter text for sentiment analysis: ")

        if text.lower() == "quit":
            break

        textblob_classification, textblob_sentiment = get_sentiment_textblob(text, THRESHOLD_TEXTBLOB)
        vader_classification, vader_sentiment = get_sentiment_vader(text, THRESHOLD_VADER)

        mood = Mood(textblob_classification, vader_classification, textblob_sentiment, vader_sentiment)

        print(f"TextBlob - Classification: {mood.textblob_classification}, Sentiment: {mood.textblob_sentiment}")
        print(f"VADER - Classification: {mood.vader_classification}, Sentiment: {mood.vader_sentiment}")

if __name__ == '__main__':
    main()
