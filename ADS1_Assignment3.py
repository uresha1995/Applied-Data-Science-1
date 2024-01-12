# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 21:44:41 2024

@author: uresha

"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="KMeans is known to have a memory leak on Windows with MKL*")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import cluster
import sklearn.preprocessing as pp
import sklearn.metrics as skmet

co2_cap = pd.read_csv("CO2 per $ of GDP.csv")
print(co2_cap.describe())
print(co2_cap)

# From 1990 to 2020, data use for clustering
# Countries with one NaN drop from data set
co2_cap = co2_cap[(co2_cap["1990"].notna()) & (co2_cap["2020"].notna())]
co2_cap = co2_cap.reset_index(drop=True)

#Check types
print(co2_cap.dtypes)

# Extract the year 1990
increment = co2_cap[["Country Name", "1990"]].copy()

# Calculate the increment of co2 vs gdp over 30 years
increment["Growth"] = 100.0 / 30.0 * (co2_cap["2020"] - co2_cap["1990"]) / co2_cap["1990"]
print(increment.describe())
print()

# Identify the data type
print(increment.dtypes)

# Plot the graph
plt.figure(figsize=(10, 10))
plt.scatter(increment["1990"], increment["Growth"])

plt.xlabel("CO2 emission per unit of GDP 1990")
plt.ylabel("Growth per year %")
plt.title("CO2 emission vs GDP Growth per year") ###########
plt.show()

#Creat a scaler object
scaler = pp.RobustScaler()

#Set up the scaler object and extract the columns for clustering
df_ex = increment[["1990", "Growth"]].copy()

# Check for and replace any infinite or extremely large values
df_ex.replace([np.inf, -np.inf], np.nan, inplace=True)
df_ex.fillna(df_ex.max(), inplace=True)

scaler.fit(df_ex)

#Apply the scaling
norm = scaler.transform(df_ex)

#Plot the graph
plt.figure(figsize=(8, 8))
plt.scatter(norm[:, 0], norm[:, 1])

plt.xlabel("CO2 emission per unit of GDP 1990")
plt.ylabel("Growth per year %")
plt.title("CO2 emission vs GDP Growth per year_scaler") ##############
plt.show()
   

def silhoutte_value(xy, n):
    """ This function is use to calculate the silhoutte score
    for n clusters """
    
    #set up the clusters with the expected clusters
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)
    
    #fit the data, result are sorted in kmeans object
    kmeans.fit(xy)
    
    labels = kmeans.labels_
    
    #calculate the silhoutte score
    score = (skmet.silhouette_score(xy, labels))
    
    return score

# calculate silhouette score for 2 to 10 clusters
for ic in range(2, 11):
    score = silhoutte_value(norm, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}") 
    
#Set up the clusters with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=3, n_init=20)   

#Fit the data and stored results in the kmeans object
kmeans.fit(norm)

#Extract the cluster labels
labels = kmeans.labels_

#Extract the estimated cluster centres and convert to original scales
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]

print(df_ex)

plt.figure(figsize=(8, 8))
cm = matplotlib.colormaps["Paired"]

# plot data with kmeans cluster number
plt.scatter(increment["1990"], increment["Growth"], 10, labels, marker="o", cmap=cm)

# show cluster centres
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
plt.xlabel("CO2 emmission per head 1990")
plt.ylabel("GDP CO2 emmission increament/year [%]")
plt.show()
