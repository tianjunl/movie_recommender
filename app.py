'''
web application code and interface between front and back-end
'''

from flask import Flask, request, render_template
from utils import movies, id_to_title
from recommender import recommend_random

# instantiates a flask object with the reference point for this app being this python script
app = Flask(__name__)

# this decorator gives our function new functionality
# it now knows in which URL to redener the information
@app.route('/')
def landing_page():
    '''
    this page takes in user info via a html form

    TODO make the users input movie titles instead of IDs!
    '''
    return render_template('landing_page.html')


@app.route('/recommendations')
def recommender():
    ''' 
    this page should access the user input and transform it into recommendations
    '''
    movie_ids = request.args.getlist('movie_ids')
    ratings = request.args.getlist('ratings')
    
    query = dict(zip(movie_ids, ratings))
    
    # here you make your transformation from query to recommendation
    recs = recommend_random(query=query, movies=movies)
    titles = id_to_title(recs, movies)
    zipped = zip(recs, titles)

    return render_template('recommendations.html', query=query, zipped=zipped)


@app.route('/movies/<int:movieId>')
def movie_info(movieId):
    '''
    this page give the user info about the movie
    '''
    info = movies.set_index('movieId').loc[movieId]
    # TODO: use the search_title function in the utils.py to find the movie from the dataset

    return render_template('movie_info.html', info=info, movieId=movieId)
    

# ensures this in the '__main__' module and not imported before running the code below
if __name__ == "__main__":
    # executes the app on the development server which we can access via: http://127.0.0.1:5000/
    # debug=True restarts the server everytime we save changes so we can see them in real time
    # also gives us verbose debugging information in the terminal
    app.run(debug=True)
