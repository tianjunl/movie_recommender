import pandas as pd
import re
from scipy.sparse import csr_matrix


ratings = pd.read_csv("./data/ml-latest-small/ratings.csv")
movies = pd.read_csv("./data/ml-latest-small/movies.csv")


def get_ratings_matrix(ratings):
    """
    This function returns a CSR matrix of the given dataframe
    """
    R = csr_matrix((ratings["rating"], (ratings["userId"], ratings["movieId"])))
    return R


def id_to_title(ids):
    """
    This function takes in movie ids and returns movie titles.
    """
    value_exist = []
    for id in ids:
        value_exist.append(id in movies["movieId"].unique())

    if sum(value_exist) == len(ids):
        return movies.set_index("movieId").loc[ids]["title"].to_list()
    else:
        ids = [i for (i, value) in zip(ids, value_exist) if value]
        return movies.set_index("movieId").loc[ids]["title"].to_list()


def title_to_id(titles):
    """
    This function takes in complete movie titles and returns movie ids.
    """
    return movies.set_index("title").loc[titles]["movieId"].to_list()


def search_movies(search_words, movies):
    """
    This function takes in key words, and returns a dataframe 
    of related movies and a status value. The status value is 1
    if the search_result is not empty, otherwise 0.
    """
    search_words = re.sub(
        r"[\n\:\,\.\?\!\*\$\#\(\)]|\'\w+\b|\bof\b|\bthe\b|\ba\b|\ban\b|\bto\b",
        " ",
        search_words.lower(),
    )
    search_words = search_words.split()

    movie_titles = movies["title"].str.lower()
    matches = pd.DataFrame({"movieId": [], "title": [], "genres": []})
    for word in search_words:
        condition = movie_titles.str.contains(word)
        matches = pd.concat([matches, movies[condition]], axis=0)
    matches = (
        matches.groupby(matches.columns.tolist())
        .size()
        .reset_index()
        .rename(columns={0: "match"})
    )
    if matches.shape[0] >= 1:
        matches["movieId"] = matches["movieId"].astype(int)
        found_status = 1
    else:
        found_status = 0
    return found_status, matches.sort_values("match", ascending=False)


def best_match(search_df):
    """
    This function takes in search result dataframe and 
    returns the movie ID and title the best match.
    """
    closest_title = search_df.iloc[0,]["title"]
    closest_id = search_df.iloc[0,]["movieId"]
    return closest_id, closest_title


def probable_matches(search_df):
    """
    This function takes in search result dataframe and 
    returns the movie IDs and titles the probable matches.
    """
    probable_ids = search_df.iloc[1:,]["movieId"].to_list()
    probable_titles = search_df.iloc[1:,]["title"].to_list()
    return probable_ids, probable_titles


# filter movies with more than 20 ratings
ratings_per_movie = ratings.groupby("movieId")["userId"].count()
popular_movies = ratings_per_movie.loc[ratings_per_movie > 20].index
ratings = ratings.loc[ratings["movieId"].isin(popular_movies)]

# filter out movies with an average rating lower than 2
mean_ratings = ratings.groupby("movieId")[["rating"]].mean()
mean_ratings = mean_ratings[mean_ratings["rating"] >= 2].index
ratings = ratings.loc[ratings["movieId"].isin(mean_ratings)]

if __name__ == "__main__":
    id = [24, 36, 2860, 4305]
    names = id_to_title(id, movies)
    print(names)
