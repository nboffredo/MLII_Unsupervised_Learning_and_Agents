import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics

data = np.genfromtxt('SpeedDating.csv', dtype=float, delimiter=';', names=True, filling_values=0)

#print(data.all())
actual_data = data.view((float, len(data.dtype.names)))

test = np.unique(data["iid"])

print(test.size)


# pca = PCA().fit(actual_data)

# variance_ratio = pca.explained_variance_ratio_
# n_components = 10
# explained_variances = list()
# for k in range(n_components):
#     explained_variance = sum(variance_ratio[:k])
#     explained_variances.append(explained_variance)

# plt.plot(range(1, n_components), explained_variances[1:], "o")
# plt.title("explained variance")
# plt.xlabel("nb components")
# plt.ylabel("explained variance by the first n compoentnts")
# plt.savefig("results/explained_variance.pdf")

nbs_of_clusters = range(1, 50)

silouhette_scores = dict()
calinski_harabasz_scores = dict()
for nb_cluster in nbs_of_clusters:
    ac = AgglomerativeClustering(nb_cluster, metric="cosine", linkage='single').fit(actual_data)
    if (nb_cluster <= 1):
        continue
    labels = ac.labels_
    silouhette_scores[nb_cluster] = metrics.silhouette_score(actual_data, labels, metric='euclidean')
    calinski_harabasz_scores[nb_cluster] = metrics.calinski_harabasz_score(actual_data, labels)

ac = AgglomerativeClustering(max(silouhette_scores, key=silouhette_scores.get), metric="cosine", linkage='single').fit(actual_data)
x_data = actual_data[:, 0]
y_data = actual_data[:, 1]
plt.scatter(x_data, y_data, c=ac.labels_, cmap='rainbow')
plt.title("Agglomerative Clustering : Silouhette scores as a function of nb_cluster")
plt.xlabel("nb_cluster")
plt.ylabel("silouhette_score")
plt.savefig("results/AgglomerativeClustering_silouhette_result.pdf")
plt.close()

plt.plot(nbs_of_clusters[1:], silouhette_scores.values(), "o")
plt.title("Silouhettes scores as a function of nb_cluster")
plt.xlabel("nb_cluster")
plt.ylabel("silouhette_score")
plt.savefig(f"results/silouhettes_scores_as_a_function_of_nb_cluster_AC.pdf")
plt.close()

ac = AgglomerativeClustering(max(calinski_harabasz_scores, key=calinski_harabasz_scores.get), metric="cosine", linkage='single').fit(actual_data)
x_data = actual_data[:, 0]
y_data = actual_data[:, 1]
plt.scatter(x_data, y_data, c=ac.labels_, cmap='rainbow')
plt.title("Agglomerative Clustering : Calinski-harabasz scores as a function of nb_cluster")
plt.xlabel("nb_cluster")
plt.ylabel("calinski_harabasz_score")
plt.savefig("results/AgglomerativeClustering_calinski_harabasz_result.pdf")
plt.close()

plt.plot(nbs_of_clusters[1:], calinski_harabasz_scores.values(), "o")
plt.title("Calinski-harabasz scores as a function of nb_cluster")
plt.xlabel("nb_cluster")
plt.ylabel("calinski_harabasz_score")
plt.savefig(f"results/calinski_harabasz_scores_as_a_function_of_nb_cluster_AC.pdf")
plt.close()