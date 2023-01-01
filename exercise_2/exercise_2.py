import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tsne import tsne as TSNE

pca_data = np.load("data.npy")
pca_labels = np.load("labels.npy")

colMap = {0: "indianred", 1: "blue"}
colors = list(map(lambda x: colMap.get(x), pca_labels))
labl = {0:'calm',1:'tempest'}
alphas = {0: 0.7, 1: 0.5}

tsne = TSNE(pca_data, no_dims=3)

print(tsne)

pca2 = PCA(n_components=2)
pca3 = PCA(n_components=3)

pca2_projected = pca2.fit_transform(pca_data)
pca3_projected = pca3.fit_transform(pca_data)

pca = PCA().fit(pca_data)

variance_ratio = pca.explained_variance_ratio_
n_components = 7
explained_variances = list()
for k in range(n_components):
    explained_variance = sum(variance_ratio[:k])
    explained_variances.append(explained_variance)

plt.plot(range(1, n_components), explained_variances[1:], "o")
plt.title("explained variance")
plt.xlabel("nb components")
plt.ylabel("explained variance by the first n compoentnts")
plt.savefig("results/explained_variance.pdf")

# principal component obtained by the algorithm
print("components")
print(pca2.components_)
print(pca3.components_)

Xax = pca2_projected[:, 0]
Yax = pca2_projected[:, 1]
fig = plt.figure()
ax = fig.add_subplot()
for l in np.unique(pca_labels):
    ix=np.where(pca_labels==l)
    ax.scatter(Xax[ix], Yax[ix], s=40,
           label=labl[l], alpha=alphas[l])
ax.set_title("PCA reduction to 2 dimensions")
ax.legend()
ax.set_xlabel("First Principal Component")
ax.set_ylabel("Second Principal Component")
plt.savefig("results/pca_test_2D.pdf")

Xax = pca3_projected[:, 0]
Yax = pca3_projected[:, 1]
Zax = pca3_projected[:, 2]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for l in np.unique(pca_labels):
    ix=np.where(pca_labels==l)
    ax.scatter(Xax[ix], Yax[ix], Zax[ix], s=40,
           label=labl[l], alpha=alphas[l])
ax.set_title("PCA reduction to 3 dimensions")
ax.set_xlabel("First Principal Component", fontsize=11)
ax.set_ylabel("Second Principal Component", fontsize=11)
ax.set_zlabel("Third Principal Component", fontsize=11)
ax.legend()
plt.savefig("results/pca_test_3D.pdf")

# variance carried by those axes
print(f"\nexplained variance pca2 {pca2.explained_variance_}")
print(f"\nexplained variance pca3 {pca3.explained_variance_}")

# variance ratio carried by those axes
print(f"\nexplained variance pca2 {pca2.explained_variance_}")
print(f"\nexplained variance ratio pca2 {pca2.explained_variance_ratio_}")
print(f"\nexplained variance pca3 {pca3.explained_variance_}")
print(f"\nexplained variance ratio pca3 {pca3.explained_variance_ratio_}")