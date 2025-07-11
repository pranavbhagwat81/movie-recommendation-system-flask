import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load the dataset from same folder

csv_path = os.path.join(os.path.dirname(__file__), "imdb.csv")
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"File not found: {csv_path}")
    exit(1)

# Extract required columns
df = df[['Series_Title', 'Overview', 'Genre']]

# Data cleaning - Remove null value rows
df = df.dropna()
df = df.reset_index(drop=True)
print(f"Dataset shape after cleaning: {df.shape}")

# Create a combined column of Title, Description, and Genre
df['Combined'] = df['Series_Title'] + ' ' + df['Overview'] + ' ' + df['Genre']

# Keep only necessary columns
df = df[['Series_Title', 'Combined']]

# Remove stop words from combined column
stop_words = set(stopwords.words('english'))


def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_text = ' '.join(
        [word for word in word_tokens if word.lower() not in stop_words])
    return filtered_text


df['Combined'] = df['Combined'].apply(remove_stopwords)

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['Combined'])
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# Create cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations


def get_recommendations(title, cosine_sim=cosine_sim):
    """
    Get top 5 movie recommendations based on content similarity
    """
    # Get the index of the movie that matches the title
    matches = df[df['Series_Title'] == title]
    if matches.empty:
        # If not found, find the most similar title using string similarity
        possible_titles = df['Series_Title'].tolist()
        close_matches = get_close_matches(
            title, possible_titles, n=1, cutoff=0.6)
        if not close_matches:
            print(f"Title '{title}' not found and no similar titles found.")
            return []
        print(
            f"Title '{title}' not found. Using closest match: '{close_matches[0]}'")
        idx = df[df['Series_Title'] == close_matches[0]].index[0]
    else:
        idx = matches.index[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar movies
    sim_scores = sim_scores[1:6]

    # Get the movie indices
    movie_indices = [int(i[0]) for i in sim_scores]

    # Return the top 5 most similar movies
    return df.iloc[movie_indices]['Series_Title'] if movie_indices else []


# Example usage
if __name__ == "__main__":
    # Test the recommendation system
    movie_title = input("Enter a movie title: ")
    recommendations = get_recommendations(movie_title)

    # get if recommendations are found
    recommendations_list = list(recommendations)
    if not recommendations_list:
        print(f"No recommendations found for '{movie_title}'.")
    else:
        print(f"Top 5 recommendations for '{movie_title}':")
        for i, movie in enumerate(recommendations_list, 1):
            print(f"{i}. {movie}")
