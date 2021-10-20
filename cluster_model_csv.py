import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


moviegenres = (
    pd.read_csv("./data/movies_genres.csv")
    .rename(columns={"movieid": "movieId"})
    .set_index("movieId")
)
X = moviegenres.drop(["title", "year"], axis=1)

pca = PCA()
T = pca.fit_transform(X)

clustering = KMeans(n_clusters=18)
clustering.fit(T[:, 0:10])
moviegenres["cluster_no"] = clustering.labels_
moviegenres[["title", "cluster_no"]].to_csv("./data/movie_clusters.csv")
