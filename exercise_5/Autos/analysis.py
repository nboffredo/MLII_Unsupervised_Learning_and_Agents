import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics, preprocessing
import pandas as pd
import missingno as msno
data = pd.read_csv('autos.csv')

msno.bar(data, sort='ascending')

print("Actual data :")

data.offerType = pd.factorize(data.offerType)[0]
data.notRepairedDamage = pd.factorize(data.notRepairedDamage)[0]
data.seller = pd.factorize(data.seller)[0]
data.abtest = pd.factorize(data.abtest)[0]
data.gearbox = pd.factorize(data.gearbox)[0]

data.insert(0, 'vehicleType', data.pop('vehicleType'))
data.head()

data.drop(columns=['index', 'name','dateCrawled', 'dateCreated', 'nrOfPictures', 'lastSeen', 'monthOfRegistration', 'postalCode'], inplace=True)

print(data)