# implement K-mean
# implement hierarchical_clustering

# normalized cut
# kneed

import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics


data = np.load("data.npy")

nbs_of_clusters = range(1, 15)
silouhette_scores = dict()
calinski_harabasz_scores = dict()

for nb_cluster in nbs_of_clusters:
    kmeans = KMeans(n_clusters=nb_cluster, n_init=8, random_state=1).fit(data)
    if (nb_cluster <= 1):
        continue
    labels = kmeans.labels_
    silouhette_scores[nb_cluster] = metrics.silhouette_score(data, labels, metric='euclidean')
    calinski_harabasz_scores[nb_cluster] = metrics.calinski_harabasz_score(data, labels)

kmeans = KMeans(n_clusters= max(silouhette_scores, key=silouhette_scores.get), n_init=8, random_state=0).fit(data)
x_data = data[:, 0]
y_data = data[:, 1]
plt.plot(x_data, y_data, "o")
centroids = kmeans.cluster_centers_
x_centroids = centroids[:, 0]
y_centroids = centroids[:, 1]
plt.plot(x_centroids, y_centroids, "x", color="orange", label="centroids")
plt.savefig("results\KMean/KMean_silouhette_result.pdf")
plt.close()

# kneedle.plot_knee()
# plt.savefig("results\KMean_knee.pdf")
# plt.close()


plt.plot(nbs_of_clusters[1:], silouhette_scores.values(), "o")
plt.title("Silouhettes scores as a function of nb_cluster")
plt.xlabel("nb_cluster")
plt.ylabel("silouhette_score")
plt.savefig(f"results/KMean/silouhettes_scores_as_a_function_of_nb_cluster.pdf")
plt.close()

kmeans = KMeans(n_clusters= max(calinski_harabasz_scores, key=calinski_harabasz_scores.get), n_init=8, random_state=0).fit(data)
x_data = data[:, 0]
y_data = data[:, 1]
plt.plot(x_data, y_data, "o")
centroids = kmeans.cluster_centers_
x_centroids = centroids[:, 0]
y_centroids = centroids[:, 1]
plt.plot(x_centroids, y_centroids, "x", color="orange", label="centroids")
plt.savefig("results\KMean/KMean_calinski_harabasz_result.pdf")
plt.close()

plt.plot(nbs_of_clusters[1:], calinski_harabasz_scores.values(), "o")
plt.title("Calinski-harabasz scores as a function of nb_cluster")
plt.xlabel("nb_cluster")
plt.ylabel("calinski_harabasz_score")
plt.savefig(f"results/KMean/calinski_harabasz_scores_as_a_function_of_nb_cluster.pdf")
plt.close()

silouhette_scores = dict()
calinski_harabasz_scores = dict()
for nb_cluster in nbs_of_clusters:
    ac = AgglomerativeClustering(nb_cluster, metric="cosine", linkage='single').fit(data)
    if (nb_cluster <= 1):
        continue
    labels = ac.labels_
    silouhette_scores[nb_cluster] = metrics.silhouette_score(data, labels, metric='euclidean')
    calinski_harabasz_scores[nb_cluster] = metrics.calinski_harabasz_score(data, labels)

ac = AgglomerativeClustering(max(silouhette_scores, key=silouhette_scores.get), metric="cosine", linkage='single').fit(data)
x_data = data[:, 0]
y_data = data[:, 1]
plt.scatter(x_data, y_data, c=ac.labels_, cmap='rainbow')
plt.title("Agglomerative Clustering : Silouhette scores as a function of nb_cluster")
plt.xlabel("nb_cluster")
plt.ylabel("silouhette_score")
plt.savefig("results/AgglomerativeClustering/AgglomerativeClustering_silouhette_result.pdf")
plt.close()

plt.plot(nbs_of_clusters[1:], silouhette_scores.values(), "o")
plt.title("Silouhettes scores as a function of nb_cluster")
plt.xlabel("nb_cluster")
plt.ylabel("silouhette_score")
plt.savefig(f"results/AgglomerativeClustering/silouhettes_scores_as_a_function_of_nb_cluster_AC.pdf")
plt.close()

ac = AgglomerativeClustering(max(calinski_harabasz_scores, key=calinski_harabasz_scores.get), metric="cosine", linkage='single').fit(data)
x_data = data[:, 0]
y_data = data[:, 1]
plt.scatter(x_data, y_data, c=ac.labels_, cmap='rainbow')
plt.title("Agglomerative Clustering : Calinski-harabasz scores as a function of nb_cluster")
plt.xlabel("nb_cluster")
plt.ylabel("calinski_harabasz_score")
plt.savefig("results/AgglomerativeClustering/AgglomerativeClustering_calinski_harabasz_result.pdf")
plt.close()

plt.plot(nbs_of_clusters[1:], calinski_harabasz_scores.values(), "o")
plt.title("Calinski-harabasz scores as a function of nb_cluster")
plt.xlabel("nb_cluster")
plt.ylabel("calinski_harabasz_score")
plt.savefig(f"results/AgglomerativeClustering/calinski_harabasz_scores_as_a_function_of_nb_cluster_AC.pdf")
plt.close()