from sklearn.neighbors import NearestNeighbors
import pickle
from utils import ratings, get_ratings_matrix


R = get_ratings_matrix(ratings)

model = NearestNeighbors(metric='cosine')
model.fit(R)

with open('./distance_recommender.pkl', 'wb') as file:
    pickle.dump(model, file)
