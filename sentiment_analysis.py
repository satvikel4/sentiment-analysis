from textblob import TextBlob
from dataclasses import dataclass

@dataclass
class Mood:
    classification: str
    sentiment: float

THRESHOLD = 0.33

def get_sentiment(text: str, threshold: float) -> Mood:
    sentiment = TextBlob(text).sentiment.polarity

    if sentiment >= threshold:
        classification = "Positive"
    elif sentiment <= -threshold:
        classification = "Negative"
    else:
        classification = "Neutral"

    return Mood(classification, sentiment)

def main():
    print("Enter 'quit' to exit sentiment analysis")
    
    while True:
        text = input("Enter text for sentiment analysis: ")
        
        if text.lower() == "quit":
            break
            
        mood = get_sentiment(text, threshold=THRESHOLD)
        print(f"Classification: {mood.classification}, Sentiment: {mood.sentiment}")

if __name__ == '__main__':
    main()
