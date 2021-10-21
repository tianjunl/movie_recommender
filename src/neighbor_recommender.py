import numpy as np
import pandas as pd
import pickle
from utils import ratings, get_ratings_matrix, id_to_title


movies = pd.read_csv('./data/ml-latest-small/movies.csv')
with open('./models/distance_recommender.pkl', 'rb') as file:
    model = pickle.load(file)

R = get_ratings_matrix(ratings)


def recommend_neighborhood(query, model, ratings, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model.
    Returns a list of k movie ids.
    """
    # 1. candiate generation
    # construct a user vector
    user_vec = np.repeat(0, 168253)
    user_vec[list(query.keys())] = list(query.values())

    # 2. scoring
    # find n neighbors
    distances, userIds = model.kneighbors(
        [user_vec], n_neighbors=10, return_distance=True)
    distances = distances[0]
    userIds = userIds[0]
    neighborhood = ratings.set_index('userId').loc[userIds]
    # calculate the score with the NMF model
    scores = neighborhood.groupby('movieId')['rating'].sum()

    # 3. ranking
    # filter out movies already seen by the user
    already_seen = scores.index.isin(query.keys())
    scores.loc[already_seen] = 0
    # return the top-k highst rated movie ids or titles
    scores = scores.sort_values(ascending=False)
    recommendations = scores.head(k).index.tolist()

    return recommendations


if __name__ == "__main__":
    example_query = {
        # movieId, rating
        296: 5,
        109487: 5,
        541: 5,
        1232: 5,
        36363: 5,
        1089: 5,
        114670: 5,
        5903: 5,
        1253: 5
    }

    recommendations = recommend_neighborhood(example_query, model, ratings, 10)
    rec_titles = id_to_title(recommendations, movies)
    print(rec_titles)
