import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class PrimeRecommender:
    def __init__(self, data_path='amazon_prime_titles.csv'):
        self.data = pd.read_csv(data_path)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.similarity_matrix = None
        self._prepare_data()
        
    def _prepare_data(self):
        # Combine relevant features for content-based filtering
        self.data['combined_features'] = self.data['title'] + ' ' + self.data['description'].fillna('')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data['combined_features'])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
    def get_recommendations(self, title, n_recommendations=5):
        try:
            # Find the index of the movie
            idx = self.data[self.data['title'] == title].index[0]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.similarity_matrix[idx]))
            
            # Sort movies based on similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top n recommendations
            sim_scores = sim_scores[1:n_recommendations+1]
            movie_indices = [i[0] for i in sim_scores]
            
            # Return recommended movies
            return self.data.iloc[movie_indices][['title', 'description']].to_dict('records')
            
        except IndexError:
            return []
            
    def get_popular_movies(self, n=5):
        # Simple popularity-based recommendation
        return self.data.sort_values('rating', ascending=False).head(n)[['title', 'description']].to_dict('records')

class NetflixRecommender:
    def __init__(self, data_path='netflix_titles.csv'):
        self.data = pd.read_csv(data_path)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.similarity_matrix = None
        self._prepare_data()
        
    def _prepare_data(self):
        # Combine relevant features for content-based filtering
        self.data['combined_features'] = self.data['title'] + ' ' + self.data['description'].fillna('')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data['combined_features'])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
    def get_recommendations(self, title, n_recommendations=5):
        try:
            # Find the index of the movie
            idx = self.data[self.data['title'] == title].index[0]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.similarity_matrix[idx]))
            
            # Sort movies based on similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top n recommendations
            sim_scores = sim_scores[1:n_recommendations+1]
            movie_indices = [i[0] for i in sim_scores]
            
            # Return recommended movies with additional info
            return self.data.iloc[movie_indices][['title', 'description', 'type', 'rating', 'listed_in']].to_dict('records')
            
        except IndexError:
            return []
            
    def get_popular_movies(self, n=5):
        # Simple popularity-based recommendation using release year as proxy for popularity
        return self.data.sort_values('release_year', ascending=False).head(n)[['title', 'description', 'type', 'rating', 'listed_in']].to_dict('records') 