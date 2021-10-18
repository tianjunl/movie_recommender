import numpy as np
import pandas as pd
import pickle
from utils import ratings, get_ratings_matrix


movies = pd.read_csv('./data/ml-latest-small/movies.csv')
with open('./models/nmf_recommender.pkl', 'rb') as file:
    model_nmf = pickle.load(file)
with open('./models/distance_recommender.pkl', 'rb') as file:
    model_dist = pickle.load(file)

R = get_ratings_matrix(ratings)


def recommend_random(query, movies, k=10):
    """
    Dummy recommender that recommends a list of random movies. Ignores the actual query.
    """
    return movies.sample(k)['movieId'].to_list()


# recommender_systems_intro_filled.ipynb
def recommend_popular(query, ratings, k=10):
    """
    Filters and recommends the top k movies for any given input query. 
    Returns a list of k movie ids.
    """    
    
    return [364, 372, 43, 34, 243]

# clustering_filled.ipynb
def recommend_cluster(query, k=10):
    """
    Filters and recommends the top k movies from a cluster a given input query. 
    Returns a list of k movie ids.
    """    
    
    return [364, 372, 43, 34, 243]

# NMF_filled.ipynb
def recommend_nmf(query, model=model_nmf, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.
    """
    # 1. candiate generation
    # construct a user vector
    user_vec = np.repeat(0, 168253)
    user_vec[list(query.keys())] = list(query.values())    
   
    # 2. scoring    
    # calculate the score with the NMF model
    scores = model.inverse_transform(model.transform([user_vec]))
    scores = pd.Series(scores[0])
    
    # 3. ranking
    # filter out movies already seen by the user
    scores[list(query.keys())] = 0
    # return the top-k highst rated movie ids or titles
    scores = scores.sort_values(ascending=False)
    recommendations = scores.head(k).index.tolist()

    return recommendations


# neighborhood_based_filtering.ipynb
def recommend_neighborhood(query, model=model_dist, ratings=ratings, k=10):
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
    distances, userIds = model.kneighbors([user_vec], n_neighbors=10, return_distance=True)
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
