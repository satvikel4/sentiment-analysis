from textblob import TextBlob
from dataclasses import dataclass

@dataclass
class Mood:
    classification: str
    sentiment: float

def get_sentiment(text: str, *, threshold: float) -> Mood:
    sentiment = TextBlob(text).sentiment.polarity

    positive_threshold = threshold
    negative_threshold = -threshold

    if sentiment >= positive_threshold:
        return Mood("Positive", sentiment)
    elif sentiment <= negative_threshold:
        return Mood("Negative", sentiment)
    else:
        return Mood("Neutral", sentiment)

if __name__ == '__main__':
    print("Enter 'quit' to exit sentiment analysis")
    text = input("Enter text for sentiment analysis: ")
    while text.lower() != "quit":
        mood = get_sentiment(text, threshold=0.33)
        print("Classification: " + mood.classification + ", Sentiment: " + str(mood.sentiment))
        text = input("Enter text for sentiment analysis: ")

