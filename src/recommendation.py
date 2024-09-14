import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from math import sqrt

def load_data(filepath):
    """Load the dataset from the given file path."""
    ratings = pd.read_csv(filepath)
    return ratings

def create_user_item_matrix(ratings):
    """Create a user-item matrix from ratings data."""
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
    return user_item_matrix

def compute_user_similarity(user_item_matrix):
    """Compute cosine similarity between users."""
    user_item_matrix_filled = user_item_matrix.fillna(0)
    user_similarity = cosine_similarity(user_item_matrix_filled)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
    return user_similarity_df

def predict_ratings(user_item_matrix, user_similarity):
    """Predict ratings based on user similarity."""
    mean_user_rating = user_item_matrix.mean(axis=1)
    ratings_diff = (user_item_matrix.T - mean_user_rating).T
    pred = mean_user_rating.values[:, np.newaxis] + user_similarity.dot(ratings_diff.fillna(0)) / np.array([np.abs(user_similarity).sum(axis=1)]).T
    pred_df = pd.DataFrame(pred, index=user_item_matrix.index, columns=user_item_matrix.columns)
    return pred_df

def evaluate_model(true_ratings, predicted_ratings):
    """Evaluate the model using RMSE."""
    true_ratings_flat = true_ratings.values.flatten()
    predicted_ratings_flat = predicted_ratings.values.flatten()
    mask = ~np.isnan(true_ratings_flat)
    mse = mean_squared_error(true_ratings_flat[mask], predicted_ratings_flat[mask])
    rmse = sqrt(mse)
    return rmse

if __name__ == "__main__":
    # Load the data
    ratings = load_data('data/ratings.csv')
    user_item_matrix = create_user_item_matrix(ratings)
    
    # Compute user similarity
    user_similarity = compute_user_similarity(user_item_matrix)
    
    # Predict ratings
    predicted_ratings = predict_ratings(user_item_matrix, user_similarity)
    
    # Evaluate the model
    rmse = evaluate_model(user_item_matrix, predicted_ratings)
    print(f'Root Mean Squared Error: {rmse:.4f}')
