"""
Web application code and interface between front and back-end.
"""

from flask import Flask, request, render_template
from utils import (
    movies,
    id_to_title,
    search_movies,
    best_match,
    probable_matches,
    title_to_id,
)
from recommender import (
    recommend_random,
    recommend_popular,
    recommend_cluster,
    recommend_nmf,
    recommend_neighborhood,
)


app = Flask(__name__)


@app.route("/")
def landing_page():
    """
    This page takes in user query for recommendations
    and for searching movie id via html forms.
    """
    if "search_words" in request.args:
        search_words = request.args["search_words"]
        found_status, search_result = search_movies(search_words, movies)

        if found_status == 1:
            closest_id, closest_title = best_match(search_result)
            probable_ids, probable_titles = probable_matches(search_result)
            zip_matches = zip(probable_ids, probable_titles)

            return render_template(
                "landing_page.html",
                found_status=found_status,
                closest_id=closest_id,
                closest_title=closest_title,
                zip_matches=zip_matches,
            )
        else:
            return render_template(
                "landing_page.html",
                found_status=found_status)
    else:
        return render_template("landing_page.html")


@app.route("/recommendations")
def recommender():
    """
    This page should access the user input and transform it into recommendations
    """
    query_ids = request.args.getlist("query_ids", type=int)
    ratings = request.args.getlist("ratings", type=int)
    select_model = request.args["select_model"]
    k = request.args.get("rec_num", type=int)

    recommend = {
        "random": recommend_random,
        "popular": recommend_popular,
        "cluster": recommend_cluster,
        "NMF": recommend_nmf,
        "neighborhood": recommend_neighborhood,
    }.get(select_model, recommend_popular)
    query_titles = id_to_title(query_ids)

    if len(query_titles) == len(query_ids):
        input_valid = 1
    elif len(query_titles) > 0:
        input_valid = 0
        query_ids = title_to_id(query_titles)
    else:
        input_valid = -1

    if len(query_titles) > 0:
        query = dict(zip(query_ids, ratings))
        rec_ids = recommend(query=query, k=k)
        titles = id_to_title(rec_ids)
        zip_rec = zip(rec_ids, titles)
        query_info = zip(query_ids, query_titles)

        return render_template(
            "recommendations.html",
            input_valid=input_valid,
            query_info=query_info,
            zip_rec=zip_rec,
        )
    else:
        return render_template("recommendations.html", input_valid=input_valid)


@app.route("/movies/<int:movieId>")
def movie_info(movieId):
    """
    this page give the user info about the movie
    """
    info = movies.set_index("movieId").loc[movieId]

    return render_template("movie_info.html", info=info, movieId=movieId)


if __name__ == "__main__":
    app.run(debug=True)
