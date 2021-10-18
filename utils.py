import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


ratings = pd.read_csv('./data/ml-latest-small/ratings.csv')
movies = pd.read_csv('./data/ml-latest-small/movies.csv')


def get_ratings_matrix(ratings):
    """
    This function returns a CSR matrix of the given dataframe
    """
    R = csr_matrix((ratings['rating'], (ratings['userId'], ratings['movieId'])))
    return R

def id_to_title(ids, movies):
    """
    This function takes in movie indecies and returns movie titles.
    """
    return movies.set_index('movieId').loc[ids]['title'].to_list()


# filter movies with more than 20 ratings
ratings_per_movie = ratings.groupby('movieId')['userId'].count()
popular_movies = ratings_per_movie.loc[ratings_per_movie > 20].index
ratings = ratings.loc[ratings['movieId'].isin(popular_movies)]

# filter out movies with an average rating lower than 2
mean_ratings = ratings.groupby('movieId')[['rating']].mean()
mean_ratings = mean_ratings[mean_ratings['rating']>=2].index
ratings = ratings.loc[ratings['movieId'].isin(mean_ratings)]

if __name__=='__main__':
    id = [24, 36, 2860, 4305]
    names = id_to_title(id, movies)
    print(names)






