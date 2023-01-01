import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics, preprocessing
import pandas as pd
#data = np.genfromtxt('cars.csv', dtype=float, delimiter=';', names=True, filling_values=0)
data = pd.read_csv('cars.csv', sep=";")

print("Actual data :")

print(data)

data.drop("Car", inplace=True, axis=1)
data.drop("Cylinders", inplace=True, axis=1)
data.drop("Displacement", inplace=True, axis=1)
data.drop("Weight", inplace=True, axis=1)


data.Origin = pd.factorize(data.Origin)[0]

print(data)

corr_matrix = data.corr()
print(corr_matrix)

im = plt.imshow(corr_matrix)
plt.colorbar(im)
plt.title("correlation matrix")
plt.savefig("results/correlation.pdf")
plt.close()

#print(data.all())
#actual_data = data.view((float, len(data.dtype.names)))


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
    ac = AgglomerativeClustering(nb_cluster, metric="cosine", linkage='single').fit(data)
    if (nb_cluster <= 1):
        continue
    labels = ac.labels_
    silouhette_scores[nb_cluster] = metrics.silhouette_score(data, labels, metric='euclidean')
    calinski_harabasz_scores[nb_cluster] = metrics.calinski_harabasz_score(data, labels)

ac = AgglomerativeClustering(max(calinski_harabasz_scores, key=calinski_harabasz_scores.get), metric="cosine", linkage='single').fit_predict(data)


print(ac)

pca = PCA().fit(data)
variance_ratio = pca.explained_variance_ratio_
n_components = 7
explained_variances = list()
for k in range(n_components):
    explained_variance = sum(variance_ratio[:k])
    explained_variances.append(explained_variance)

plt.plot(range(n_components), explained_variances, "o")
plt.title("explained variance")
plt.xlabel("nb components")
plt.ylabel("explained variance by the first n compoentnts")
plt.savefig("results/explained_variance.pdf")

exit()

x_data = actual_data[:, 1]
y_data = actual_data[:, 5]
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