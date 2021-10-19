# Movie Recommender

A small online application for movie recommendations with given ratings.

Data source: [MovieLens](https://grouplens.org/datasets/movielens/) - [ml-latest-small.zip](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)




## User Instructions

- open [heroku link](https://testmovierecommender.herokuapp.com/) in the browser 

- landing page:

  - Query 1: submit movie ids and their ratings to get recommendations

  - Query 2: search a movie name to get its ID and the complete name

  - Query 2 result: 
  
    If the best match is not your movie, please scroll down to find other movies with similar names. Click on movie names to view more movie information.
  
- recommendation page:

  - the default model is `recommend_popular`

  - click on any movie name to view more movie information.

    

## Developer Instructions

1. requirements
- free heroku account
- heroku CLI installed and set up locally

2. clone the repository

```bash
git clone https://github.com/tianjunl/movie_recommender.git
cd movie_recommender
```

3. create a new heroku app

```bash
heroku create <my-app-name>`
```

4. change github (origin) connection

```bash
git remote remove origin
git remote add origin <your-github-repository>
git branch -M main
git push -u origin main
```


5. edit the content, then test the app locally

```bash
heroku local web
```


6. push code to heroku

```bash
git push heroku main
```

7. open website in browser

```bash
heroku open
```

