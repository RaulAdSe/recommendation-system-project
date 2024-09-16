## Movie Recommendation System

This project demonstrates a movie recommendation system built using collaborative filtering techniques (User-Based CF, Item-Based CF) and matrix factorization (SVD). The system analyzes user preferences and generates personalized movie recommendations.

### Objectives

- **Data Exploration**: Understand user preferences and movie properties.
- **Recommendation Algorithms**: Implement and evaluate multiple recommendation algorithms.
- **Visualizations**: Provide insights into the recommendations using visualizations.

### Data

- **Ratings Dataset**: Contains user ratings for various movies.
- **Movies Dataset**: Contains movie information including genres.

### Methodology

1. **Data Loading and Preprocessing**: 
   - Load movie and rating data, clean and prepare it for analysis.
2. **Exploratory Data Analysis (EDA)**:
   - Understand user preferences by visualizing ratings and genres.
3. **Model Implementation**:
   - **User-Based Collaborative Filtering**: Calculates user similarity using cosine similarity and predicts ratings based on similar users' ratings.
   - **Item-Based Collaborative Filtering**: Computes item similarity using genre-based TF-IDF values and recommends items similar to those the user has already rated.
   - **SVD (Matrix Factorization)**: Decomposes the user-item matrix to uncover latent features that predict missing ratings.

### Comparison of Models

- Evaluated the performance of each model using RMSE to understand which approach provides the most accurate predictions.

   ![Model Performance Comparison](path/to/model_performance_comparison_screenshot.png)

### Usage

To explore the full analysis and results, check out the [Demo Notebook](path/to/demo_notebook.ipynb) in this repository.

### How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/recommendation-system.git
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   ```bash
   jupyter notebook notebooks/demo_notebook.ipynb
   ``
