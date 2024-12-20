import numpy as np
import pandas as pd
import ast
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import kaggle

# Set up Kaggle API credentials
os.environ['KAGGLE_USERNAME'] = 'anuvanshgundev'
os.environ['KAGGLE_KEY'] = '0f3136c3ad03fcada849a5bb93457b3e'

# Define Kaggle dataset and file names
dataset = 'kartik1806/at-the-movies-dataset'
movies_file = 'tmdb_5000_movies.csv'
credits_file = 'tmdb_5000_credits.csv'

# Download the dataset using Kaggle API
os.makedirs('data', exist_ok=True)
kaggle.api.dataset_download_file(dataset, movies_file, path='data')
kaggle.api.dataset_download_file(dataset, credits_file, path='data')

# Unzip the downloaded files
import zipfile

movies_zip_path = f'data/{movies_file}.zip'
credits_zip_path = f'data/{credits_file}.zip'

if os.path.exists(movies_zip_path):
    with zipfile.ZipFile(movies_zip_path, 'r') as zip_ref:
        zip_ref.extractall('data')

if os.path.exists(credits_zip_path):
    with zipfile.ZipFile(credits_zip_path, 'r') as zip_ref:
        zip_ref.extractall('data')

# Load the datasets
movies = pd.read_csv(f'data/{movies_file}')
credits = pd.read_csv(f'data/{credits_file}')

# Merge and clean the datasets
movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

# Helper functions
def convert(text):
    return [i['name'] for i in ast.literal_eval(text)]

def convert3(text):
    return [i['name'] for i in ast.literal_eval(text)[:3]]

def fetch_director(text):
    return [i['name'] for i in ast.literal_eval(text) if i['job'] == 'Director']

def collapse(L):
    return [i.replace(" ", "") for i in L]

# Apply transformations
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Create the tags column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new = movies[['movie_id', 'title']]
new['tags'] = movies['tags'].apply(lambda x: " ".join(x))

# Vectorization and similarity calculation
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new['tags']).toarray()
similarity = cosine_similarity(vector)

# Save the pickle files
os.makedirs('model', exist_ok=True)
pickle.dump(new, open('model/movie_list.pkl', 'wb'))
pickle.dump(similarity, open('model/similarity.pkl', 'wb'))
