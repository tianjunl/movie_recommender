import numpy as np
import pandas as pd
import pickle
from utils import ratings, movies


movieclusters = pd.read_csv("./data/movie_clusters.csv", index_col="movieId")
with open("./models/nmf_recommender.pkl", "rb") as file:
    model_nmf = pickle.load(file)
with open("./models/distance_recommender.pkl", "rb") as file:
    model_dist = pickle.load(file)


def recommend_random(query, model=None, k=10):
    """
    Dummy recommender that recommends a list of random movies. Ignores the actual query.
    """
    return movies.sample(k)["movieId"].to_list()


def recommend_popular(query, model=None, k=10):
    """
    Filters and recommends the top k movies for any given input query. 
    Returns a list of k movie ids.
    """
    watched = movies[movies["movieId"].isin(list(query.keys()))]
    watched["genres"].str.split(pat="|")
    watched_genres = watched["genres"].str.split(pat="|").explode()
    watched_genres = (
        watched[["movieId"]]
        .join(watched_genres)
        .groupby("genres")
        .count()
        .rename(columns={"movieId": "count"})
        .sort_values("count", ascending=False)
    )
    query_genres = watched_genres.iloc[0:3].index.tolist()

    popularity = (
        ratings.groupby("movieId")[["rating"]]
        .count()
        .sort_values("rating", ascending=False)
        .reset_index()
        .rename(columns={"rating": "rated"})
    )

    genres = movies["genres"].str.split(pat="|").explode()
    genres = movies[["movieId"]].join(genres)
    select_genre = (
        genres[genres["genres"].isin(query_genres)]
        .merge(popularity, on="movieId")
        .sort_values("rated", ascending=False)
    )

    output = (
        select_genre.groupby(["movieId"])
        .size()
        .reset_index()
        .rename(columns={0: "match"})
        .merge(popularity, on="movieId")
        .sort_values(["match", "rated"], ascending=False)
    )
    return output["movieId"].head(k).tolist()


def recommend_cluster(query, model=None, k=10):
    """
    Filters and recommends the top k movies from a cluster a given input query. 
    Returns a list of k movie ids.
    """
    mean_ratings = (
        ratings.groupby("movieId")[["rating"]].mean().rename(columns={"rating": "mean"})
    )
    mean_ratings = pd.concat([movieclusters, mean_ratings], axis=1)

    liked = [id for id in query.keys() if query[id] >= 3]
    watched = movieclusters[movieclusters.index.isin(liked)]
    rec_cluster = watched["cluster_no"].mode().tolist()
    output = movieclusters[movieclusters["cluster_no"].isin(rec_cluster)].sort_values(
        "mean", ascending=False
    )

    return output.head(k).index.tolist()


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

    return scores.head(k).index.tolist()


def recommend_neighborhood(query, model=model_dist, k=10):
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
        [user_vec], n_neighbors=10, return_distance=True
    )
    distances = distances[0]
    userIds = userIds[0]
    neighborhood = ratings.set_index("userId").loc[userIds]
    # calculate the score with the NMF model
    scores = neighborhood.groupby("movieId")["rating"].sum()

    # 3. ranking
    # filter out movies already seen by the user
    already_seen = scores.index.isin(query.keys())
    scores.loc[already_seen] = 0
    # return the top-k highst rated movie ids or titles
    scores = scores.sort_values(ascending=False)

    return scores.head(k).index.tolist()
