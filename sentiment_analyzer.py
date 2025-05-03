from textblob import TextBlob
import pandas as pd

class PrimeSentimentAnalyzer:
    def __init__(self, data_path='amazon_prime_titles.csv'):
        self.data = pd.read_csv(data_path)
        
    def analyze_sentiment(self, title):
        try:
            # Find the movie
            movie = self.data[self.data['title'] == title].iloc[0]
            description = movie['description']
            
            # Perform sentiment analysis
            analysis = TextBlob(description)
            sentiment_score = analysis.sentiment.polarity
            
            # Classify sentiment
            if sentiment_score > 0.3:
                sentiment = "Positive 😊"
                color = "success"
            elif sentiment_score < -0.3:
                sentiment = "Negative 😞"
                color = "danger"
            else:
                sentiment = "Neutral 😐"
                color = "warning"
            
            # Get subjectivity
            subjectivity = analysis.sentiment.subjectivity
            if subjectivity > 0.6:
                subjectivity_text = "Highly Subjective"
            elif subjectivity > 0.3:
                subjectivity_text = "Moderately Subjective"
            else:
                subjectivity_text = "Objective"
            
            return {
                'title': title,
                'description': description,
                'sentiment': sentiment,
                'sentiment_score': round(sentiment_score, 2),
                'subjectivity': round(subjectivity, 2),
                'subjectivity_text': subjectivity_text,
                'color': color
            }
            
        except IndexError:
            return None 