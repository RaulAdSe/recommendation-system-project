import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.sparse.linalg import svds

# Load the ratings data from a CSV file
def load_data(filepath):
    """Load the dataset from the given file path."""
    ratings = pd.read_csv(filepath)
    return ratings

# Create a user-item matrix from the ratings data
def create_user_item_matrix(ratings):
    """Create a user-item matrix from ratings data."""
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
    return user_item_matrix

# Compute user similarity using cosine similarity
def compute_user_similarity(user_item_matrix):
    """Compute cosine similarity between users."""
    user_item_matrix_filled = user_item_matrix.fillna(0)
    user_similarity = cosine_similarity(user_item_matrix_filled)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
    return user_similarity_df

# Compute TF-IDF for content-based filtering
def compute_tf_idf(movies):
    """
    Compute TF-IDF values for each movie.

    Args:
    - movies (DataFrame): The movies data containing genres.

    Returns:
    - tf_idf (DataFrame): A DataFrame of TF-IDF values for each movie.
    """
    # Step 1: Compute Term Frequency (TF)
    # Split genres into individual terms
    movies['genres'] = movies['genres'].fillna('').apply(lambda x: x.split('|'))

    # Create a dictionary to store TF values
    tf_dict = {}

    # Calculate TF for each genre in each movie
    for idx, row in movies.iterrows():
        genre_count = len(row['genres'])
        tf_dict[row['movieId']] = {genre: row['genres'].count(genre) / genre_count for genre in row['genres']}

    # Convert the dictionary to a DataFrame
    tf = pd.DataFrame(tf_dict).T.fillna(0)

    # Step 2: Compute Inverse Document Frequency (IDF)
    # Flatten the genres list to count occurrences across all movies
    all_genres = [genre for sublist in movies['genres'] for genre in sublist]

    # Count the number of movies containing each genre
    genre_counts = pd.Series(all_genres).value_counts()

    # Compute IDF for each genre
    total_movies = len(movies)
    idf = np.log(total_movies / genre_counts)

    # Step 3: Compute TF-IDF
    # Multiply TF by IDF
    tf_idf = tf.mul(idf, axis=1)

    return tf_idf

# Predict ratings using collaborative filtering
def predict_ratings(user_item_matrix, user_similarity):
    """Predict ratings based on user similarity."""
    mean_user_rating = user_item_matrix.mean(axis=1)
    ratings_diff = (user_item_matrix.T - mean_user_rating).T
    pred = mean_user_rating.values[:, np.newaxis] + user_similarity.dot(ratings_diff.fillna(0)) / np.array([np.abs(user_similarity).sum(axis=1)]).T
    pred_df = pd.DataFrame(pred, index=user_item_matrix.index, columns=user_item_matrix.columns)
    return pred_df

# Perform Matrix Factorization using SVD
def svd_recommendation(user_item_matrix, k=50):
    """
    Perform Matrix Factorization using Singular Value Decomposition (SVD).
    
    Args:
    - user_item_matrix (DataFrame): The user-item matrix.
    - k (int): Number of singular values and vectors to compute.

    Returns:
    - predicted_ratings (DataFrame): Predicted ratings after matrix factorization.
    """
    user_item_matrix_filled = user_item_matrix.fillna(0)

    # Convert to numpy matrix
    user_ratings_matrix = user_item_matrix_filled.values

    # Perform SVD
    u, sigma, vt = svds(user_ratings_matrix, k=k)

    # Convert sigma to diagonal matrix
    sigma = np.diag(sigma)

    # Compute predicted ratings
    predicted_ratings_matrix = np.dot(np.dot(u, sigma), vt)

    # Convert to DataFrame
    predicted_ratings = pd.DataFrame(predicted_ratings_matrix, index=user_item_matrix.index, columns=user_item_matrix.columns)

    return predicted_ratings

# Evaluate the recommendation system using RMSE
def evaluate_model(true_ratings, predicted_ratings):
    """Evaluate the model using RMSE."""
    true_ratings_flat = true_ratings.values.flatten()
    predicted_ratings_flat = predicted_ratings.values.flatten()
    mask = ~np.isnan(true_ratings_flat)
    mse = mean_squared_error(true_ratings_flat[mask], predicted_ratings_flat[mask])
    rmse = sqrt(mse)
    return rmse

# Recommend items to a user based on item similarity. To compute item similarity based on genres, we can use a Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer, which is a common technique in natural language processing for text similarity.
def recommend_items(user_id, user_item_matrix, item_similarity, num_recommendations=5):
    """
    Recommend items to a user based on item similarity.

    Args:
    - user_id (int): The user ID for whom to generate recommendations.
    - user_item_matrix (DataFrame): The user-item matrix.
    - item_similarity (DataFrame): The item similarity matrix.
    - num_recommendations (int): Number of recommendations to generate.

    Returns:
    - recommendations (list): List of recommended item IDs.
    """
    # Get the user's ratings
    user_ratings = user_item_matrix.loc[user_id]

    # Align the indices of user ratings with the item similarity matrix
    user_ratings = user_ratings.reindex(item_similarity.index)

    # Compute the weighted average of item similarities by user's ratings
    scores = item_similarity.dot(user_ratings.fillna(0)).div(item_similarity.sum(axis=1))

    # Drop items already rated by the user
    scores = scores[user_ratings.isna()]

    # Get the top N recommended items
    recommended_items = scores.nlargest(num_recommendations).index.tolist()

    return recommended_items

if __name__ == "__main__":
    # Load the data
    ratings = load_data('data/ratings.csv')
    movies = load_data('data/movies.csv')
    user_item_matrix = create_user_item_matrix(ratings)
    
    # Compute user similarity
    user_similarity = compute_user_similarity(user_item_matrix)
    
    # Predict ratings using collaborative filtering
    predicted_ratings_cf = predict_ratings(user_item_matrix, user_similarity)
    
    # Predict ratings using SVD
    predicted_ratings_svd = svd_recommendation(user_item_matrix, k=50)
    
    # Compute item similarity using content-based filtering
    tf_idf = compute_tf_idf(movies)
    item_similarity = cosine_similarity(tf_idf)
    item_similarity_df = pd.DataFrame(item_similarity, index=movies['movieId'], columns=movies['movieId'])
    
    # Evaluate both models
    rmse_cf = evaluate_model(user_item_matrix, predicted_ratings_cf)
    rmse_svd = evaluate_model(user_item_matrix, predicted_ratings_svd)
    
    print(f'Collaborative Filtering RMSE: {rmse_cf:.4f}')
    print(f'SVD-Based RMSE: {rmse_svd:.4f}')
    print('Content-based item similarity computed successfully.')
    
    # Recommend items for a specific user
    user_id = 1  # Example user ID
    recommendations = recommend_items(user_id, user_item_matrix, item_similarity_df)
    print(f'Recommended items for User {user_id}: {recommendations}')
