import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
import pickle
from utils import id_to_title


movies = pd.read_csv('./data/ml-latest-small/movies.csv')

with open('./models/nmf_recommender.pkl', 'rb') as file:
    model = pickle.load(file)

def recommend_nmf(query, model, k=10):
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

if __name__ == "__main__":
    query = {
        # movieId, rating
        4470:5, 
        48:5,
        594:5,
        27619:5,
        152081:5,
        595:5,
        616:5,
        1029:5
    }
    # for testing the recommender after getting some recommendations
    relevant_items = [
        596, 4016, 1033, 134853, 
        2018, 588, 364, 26999, 75395,2085, 
        1907, 2078, 1032, 177765   
    ]

    recommendations = recommend_nmf(query, model, 10)
    rec_titles = id_to_title(recommendations, movies)
    print(rec_titles)
    print(f'list is in recom: {pd.Series(relevant_items).isin(recommendations).mean()}')
    print(f'recom is in list: {pd.Series(recommendations).isin(relevant_items).mean()}')

