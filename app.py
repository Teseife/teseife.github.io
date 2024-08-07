from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import mysql.connector
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.spatial.distance import euclidean

app = Flask(__name__)

# Establish the connection to the database
conn = mysql.connector.connect( host="localhost", user="root", password="abc1234", database="finalproject" )

# Load data into DataFrames
additional_data_df = pd.read_sql('SELECT * FROM additional_data', conn)
genres_df = pd.read_sql('SELECT * FROM genres', conn)
keywords_df = pd.read_sql('SELECT * FROM keywords', conn)
movies_df = pd.read_sql('SELECT * FROM movies', conn)
votes_df = pd.read_sql('SELECT * FROM votes', conn)

# Merge all the data into a single DataFrame
merged_df = movies_df.merge(additional_data_df, on='movie_id', how='left')
merged_df = merged_df.merge(genres_df, on='movie_id', how='left')
merged_df = merged_df.merge(keywords_df, left_on='movie_id', right_on='movie_id', how='left')
merged_df = merged_df.merge(votes_df, on='movie_id', how='left')

# Fill missing values
merged_df.fillna('NULL', inplace=True)

# Create a feature matrix
merged_df['genres'] = merged_df.groupby('movie_id')['genre'].transform(lambda x: ','.join(x))
merged_df['keywords'] = merged_df.groupby('movie_id')['keyword'].transform(lambda x: ','.join(x))
merged_df.drop_duplicates('movie_id', inplace=True)

# Create binary encodings for genres and keywords
genre_binarizer = MultiLabelBinarizer()
keyword_binarizer = MultiLabelBinarizer()

# genre_matrix = genre_binarizer.fit_transform(merged_df['genres'].str.split(','))
keyword_matrix = keyword_binarizer.fit_transform(merged_df['keywords'].str.split(','))

# Combine all features into a single matrix
feature_matrix = np.hstack([keyword_matrix])


def find_movie_index(movies_df, movie_title):
    try:
        movie_index = movies_df[movies_df['title'].str.contains(movie_title, case=False, na=False)].index[0]
        return movie_index
    except IndexError:
        return None


def calculate_similarity(matrix, target_index):
    distances = []
    for i in range(len(matrix)):
        if i != target_index:
            distance = euclidean(matrix[target_index], matrix[i])
            distances.append((i, distance))
    distances.sort(key=lambda x: x[1])
    return distances


def calculate_popularity(votes_df, top_similar_movies):
    max_votes = votes_df['vote_count'].max()
    votes_df['score'] = votes_df.apply(lambda x: x['vote_average'] ** 2 * x['vote_count'] / max_votes, axis=1)
    top_movies = votes_df.loc[top_similar_movies, 'score'].nlargest(5).index.tolist()
    return top_movies


def recommend_movies(merged_df, top_movie_indices):
    recommendations = merged_df.iloc[top_movie_indices][['title', 'links', 'vote_average', 'vote_count']]
    recommendations['image'] = recommendations['links']  # Assuming 'links' contain image URLs
    return recommendations.to_dict('records')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    favorite_movie = request.form['favorite_movie']
    movie_index = find_movie_index(movies_df, favorite_movie)

    if movie_index is None:
        return render_template('index.html', recommendations=[], error="Movie not found in the dataset.")

    similar_movies = calculate_similarity(feature_matrix, movie_index)[:20]  # Top 20 similar movies
    similar_movie_indices = [idx for idx, _ in similar_movies]
    top_movies = calculate_popularity(votes_df, similar_movie_indices)
    recommendations = recommend_movies(merged_df, top_movies)

    return render_template('index.html', recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)
