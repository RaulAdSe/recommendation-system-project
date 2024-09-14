import unittest
import pandas as pd
from src.recommendation import load_data, create_user_item_matrix, compute_user_similarity, predict_ratings

class TestRecommendationSystem(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.sample_data = {'userId': [1, 1, 2, 2, 3], 
                            'movieId': [10, 20, 10, 30, 20], 
                            'rating': [4, 5, 5, 3, 4]}
        self.ratings = pd.DataFrame(self.sample_data)

    def test_create_user_item_matrix(self):
        user_item_matrix = create_user_item_matrix(self.ratings)
        self.assertEqual(user_item_matrix.shape, (3, 3))

    def test_compute_user_similarity(self):
        user_item_matrix = create_user_item_matrix(self.ratings)
        user_similarity = compute_user_similarity(user_item_matrix)
        self.assertEqual(user_similarity.shape, (3, 3))

    def test_predict_ratings(self):
        user_item_matrix = create_user_item_matrix(self.ratings)
        user_similarity = compute_user_similarity(user_item_matrix)
        predicted_ratings = predict_ratings(user_item_matrix, user_similarity)
        self.assertEqual(predicted_ratings.shape, (3, 3))

if __name__ == '__main__':
    unittest.main()
