import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon if not already installed
nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def detect_sentiment(user_message):
    """Detect sentiment of a user message using VADER."""
    scores = sia.polarity_scores(user_message)
    compound = scores['compound']
    
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"

def chatbot_response(user_message):
    """Generate chatbot response based on sentiment."""
    sentiment = detect_sentiment(user_message)
    
    if sentiment == "positive":
        return "I'm glad to hear that! 😊 Your positivity means a lot."
    elif sentiment == "negative":
        return "I'm sorry you're feeling this way. 💙 Let me try to help."
    else:  # neutral
        return "Got it. 👍 Let’s continue with the information you need."
    
# Example interaction loop
if __name__ == "__main__":
    print("Chatbot with Sentiment Analysis (type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Chatbot: Goodbye! 👋")
            break
        response = chatbot_response(user_input)
        print(f"Chatbot: {response}")
