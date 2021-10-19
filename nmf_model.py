from sklearn.decomposition import NMF
import pickle
from utils import ratings, get_ratings_matrix


R = get_ratings_matrix(ratings)

model = NMF(n_components=55, init='nndsvd', max_iter=10000, tol=0.001, verbose=2)
model.fit(R)

with open('./models/nmf_recommender.pkl', 'wb') as file:
    pickle.dump(model, file)
