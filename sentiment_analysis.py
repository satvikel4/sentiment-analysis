from textblob import TextBlob
from dataclasses import dataclass

@dataclass
class Mood:
    classification: str
    sentiment: float

def get_sentiment(text: str, threshold: float) -> Mood:
    sentiment = TextBlob(text).sentiment.polarity

    if sentiment >= threshold:
        classification = "Positive"
    elif sentiment <= -threshold:
        classification = "Negative"
    else:
        classification = "Neutral"

    return Mood(classification, sentiment)

if __name__ == '__main__':
    print("Enter 'quit' to exit sentiment analysis")
    
    while True:
        text = input("Enter text for sentiment analysis: ")
        
        if text.lower() == "quit":
            break
            
        mood = get_sentiment(text, threshold=0.33)
        print(f"Classification: {mood.classification}, Sentiment: {mood.sentiment}")
