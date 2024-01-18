# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 21:44:41 2024

@author: uresha

"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="KMeans is known to have a memory leak on Windows with MKL")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import cluster
import sklearn.preprocessing as pp
import sklearn.metrics as skmet
import seaborn as sns
import scipy.optimize as opt
from scipy.optimize import curve_fit

co2_cap = pd.read_csv(r'C:\Users\uresha\Dropbox\PC\Desktop\UH\2. Applied Data Science 1\Assignment 3\fin\CO2 per $ of GDP.csv')
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
plt.ylabel("GDP growth per year %")
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
plt.figure(figsize=(10, 10))
plt.scatter(norm[:, 0], norm[:, 1])

plt.xlabel("CO2 emission per unit of GDP 1990")
plt.ylabel("GDP growth per year %")
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

# Plot data with kmeans cluster number
plt.scatter(increment["1990"], increment["Growth"], s=30, c=labels, cmap=cm, marker="o")

# Show cluster centres with different marker and larger size
plt.scatter(xkmeans, ykmeans, s=50, c="k", marker="D", label="Cluster Centers")

plt.xlabel("CO2 emission per head 1990")
plt.ylabel("GDP increment per year [%]")
plt.legend()
plt.show()

##DATA FITTING

def read_data():
    """
    Read and return the data File
    """
    original_dataset = pd.read_csv(
        r'C:\Users\uresha\Dropbox\PC\Desktop\UH\2. Applied Data Science 1\Assignment 3\fin\GDP per Capita.csv')
    return original_dataset


dataset = read_data()
print(dataset)

# Read and transpose the data
def transpose_and_clean_dataset(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Preparing dataset
    columns_to_remove = ['Indicator Code', 'Country Code']
    df = df.drop(columns=columns_to_remove)

    # Transpose the years to a 'Year' column
    df = df.melt(id_vars=['Indicator Name', 'Country Name'],
                 var_name='Year', value_name='Value')

    # Set 'Year' as the new index
    df = df.set_index('Year')

    return df


# Actual CSV file
result_df = transpose_and_clean_dataset(
    r'C:\Users\uresha\Dropbox\PC\Desktop\UH\2. Applied Data Science 1\Assignment 3\fin\GDP per Capita.csv')

# Print DataFrame
print(result_df)

# Save the transposed DataFrame to a new CSV file
result_df.to_csv(
    r'C:\Users\uresha\Dropbox\PC\Desktop\UH\2. Applied Data Science 1\Assignment 3\fin\transposed_dataset.csv')

# Explore the statistical properties
countries = dataset['Country Name'].unique()

# Convert the 'Year' index to a column
result_df.reset_index(inplace=True)

# Convert the 'Year' column to a numeric type
result_df['Year'] = pd.to_numeric(result_df['Year'].str.extract(r'(\d+)')[0])

# Plotting the data
plt.figure(figsize=(10, 8))
sns.lineplot(x='Year', y='Value', data=result_df, ci=None)
plt.title('GDP per Capita Over the Past Years')
plt.xlabel('Year')
plt.ylabel('GDP per Capita ($)')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

info = result_df.iloc[1:20]
sns.lineplot(x='Year', y='Value', data=result_df, ci=None)
plt.show()

#Exponential function
def exp(t, n0, g):
    """
    Calculates exponential function with scale factor n0 and growth rate g.
    """
    
    t = t - 1990
    f = n0 * np.exp(g * t)
    return f

# Check for NaN and inf values in the dataset
    """Since array must not contain infs or NaNs. Nan and inf values replace by 0 
    """
    
nan_mask = np.isnan(result_df["Value"])
inf_mask = np.isinf(result_df["Value"])

# Replace NaN and inf values 
result_df["Value"][nan_mask] = 0
result_df["Value"][inf_mask] = 0

# Parameter estimation using curve_fit
param, covar = opt.curve_fit(
    exp, result_df["Year"], result_df["Value"], p0=[1.2e12, 0.03])

# Print the results
print("GDP 1990:", param[0] / 1e9)
print("Growth rate:", param[1])

"""Since this is not a good way. Since the year-by-year data are highly correlated.
       Let curve_fit find the errors."""

# Plotting the exponential function along with the data
plt.figure(figsize=(10, 8))

# Plot the original data
sns.lineplot(x='Year', y='Value', data=result_df, ci=None)

# Plot the fitted exponential function
plt.plot(result_df["Year"], exp(result_df["Year"], *param),
         label='Trail data fit', linestyle='--')

plt.title('GDP per Capita Over the Past Years')
plt.xlabel('Year')
plt.ylabel('GDP per Capita ($)')
plt.legend(loc='upper left', bbox_to_anchor=(0, 0, 1, 1))
plt.show()



def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0, growth rate g, and inflection point t0."""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f

# Check for NaN and inf values in the dataset
nan_mask = np.isnan(result_df["Value"])
inf_mask = np.isinf(result_df["Value"])

# Replace NaN and inf values 
result_df["Value"][nan_mask] = 0
result_df["Value"][inf_mask] = 0

# Parameter estimation using curve_fit for the exponential function
param_exp, covar_exp = curve_fit(exp, result_df["Year"], result_df["Value"], p0=[1.2e12, 0.03])

#Using curve_fit for the logistic function
def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0, growth rate g, and inflection point t0."""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f

def plot_gdp_with_fits(result_df):
    # Check for NaN and inf values in the dataset
    nan_mask = np.isnan(result_df["Value"])
    inf_mask = np.isinf(result_df["Value"])

    # Replace NaN and inf values 
    result_df["Value"][nan_mask] = 0
    result_df["Value"][inf_mask] = 0

    # Parameter estimation using curve_fit for the exponential function
    param_exp, covar_exp = curve_fit(exp, result_df["Year"], result_df["Value"], p0=[1.2e12, 0.03])

    # Parameter estimation using curve_fit for the logistic function
    param_logistic, covar_logistic = curve_fit(logistic, result_df["Year"], result_df["Value"], p0=[3e12, param_exp[1], 1990])

    # Print the exponential and logistic function parameter results
    print("Logistic Initial GDP:", param_logistic[0] / 1e9)
    print("Logistic Growth rate:", param_logistic[1])
    print("Logistic Inflection point:", param_logistic[2])

    
    plt.figure(figsize=(10, 8))

    # Plot the original data
    sns.lineplot(x='Year', y='Value', data=result_df, ci=None, label='GDP')

    # Plot the fitted logistic function
    plt.plot(result_df["Year"], logistic(result_df["Year"], *param_logistic),
             label='Logistic fit', linestyle='-')

    plt.title('GDP per Capita Over the Past Years')
    plt.xlabel('Year')
    plt.ylabel('GDP per Capita ($)')
    plt.legend(loc='upper left', bbox_to_anchor=(0, 0, 1, 1))
    plt.show()

#Plot graph
plot_gdp_with_fits(result_df)


def logistic_offset(t, n0, g, t0, offset):
    """Calculates the logistic function with an offset."""
    f = offset + n0 / (1 + np.exp(-g * (t - t0)))
    return f

# Assuming result_df is defined somewhere in your code
# Adding an offset parameter (p0=[initial_GDP, growth_rate, inflection_point, offset])
param, covar = curve_fit(logistic_offset, result_df["Year"], result_df["Value"], p0=[1e12, 0.5, 1990, 5000])

# Plot the original data
sns.lineplot(x='Year', y='Value', data=result_df, ci=None, label='GDP')

# Plot the fitted logistic curve with the adjusted starting point
plt.plot(result_df["Year"], logistic_offset(result_df["Year"], *param), label='Logistic Fit', linestyle='--', color='red')

# Plot graph
plt.title('GDP per Capita Over Past Years with Logistic Fit')
plt.xlabel('Year')
plt.ylabel('GDP per Capita Growth')
plt.legend(loc='upper left', bbox_to_anchor=(0, 0, 1, 1))
plt.show()

# Extract variances and take square root to get sigmas
var = np.diag(covar)
sigma = np.sqrt(var)
print(f"Turning point {param[2]: 6.1f} +/- {sigma[2]: 4.1f}")
print(f"GDP at turning point {param[0]: 7.3e} +/- {sigma[0]: 7.3e}")
print(f"Growth rate {param[1]: 6.4f} +/- {sigma[1]: 6.4f}")


forecast_years = np.linspace(1990, 2030, 100)

# Use the logistic function with the fitted parameters to generate the forecast
forecast_values = logistic_offset(forecast_years, *param)

# Plot graph
plt.figure(figsize=(15, 10))
sns.lineplot(x=result_df["Year"], y=result_df["Value"], label="GDP",  ci=None)
sns.lineplot(x=forecast_years, y=forecast_values, label="Forecast")
plt.xlabel("Year")
plt.ylabel("GDP per Capita Growth")
plt.legend()
plt.title('GDP per Capita Forecast')
plt.show()


#Create an array and forecast for next 10 year
def deriv(x, func, parameter, i, h=1e-5):
    """
    Calculate the numerical derivative of a function with respect to the
    i-th parameter.
    """
    params_plus_h = parameter.copy()
    params_plus_h[i] += h
    deriv = (func(x, *params_plus_h) - func(x, *parameter)) / h
    return deriv

def error_prop(x, func, parameter, covar):
    """
    Calculates 1 sigma error ranges for a number or array using error
    propagation with variances and covariances taken from the covar matrix.
    Derivatives are calculated numerically.
    """
    var = np.zeros_like(x)   # initialize variance vector
    
    # Nested loop over all combinations of the parameters
    for i in range(len(parameter)):
        # derivative with respect to the ith parameter
        deriv1 = deriv(x, func, parameter, i)

        for j in range(len(parameter)):
            # derivative with respect to the jth parameter
            deriv2 = deriv(x, func, parameter, j)

            # covariance matrix element
            covar_ij = covar[i, j]

            # add to the variance vector
            var += deriv1 * covar_ij * deriv2

    sigma = np.sqrt(var)
    return sigma

year = np.linspace(1990, 2030, 100)
forecast = logistic_offset(year, *param)

# Using error_prop for error propagation
sigma = error_prop(year, logistic_offset, param, covar)
up = forecast + sigma
low = forecast - sigma

# Plot graph
plt.figure(figsize=(15, 10))
sns.lineplot(x=result_df["Year"], y=result_df["Value"], label="GDP", ci=None)
sns.lineplot(x=year, y=forecast, label="Forecast", ci=None)
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("Years")
plt.ylabel("GDP per Capita Growth")
plt.legend(loc='upper left', bbox_to_anchor=(0, 0, 1, 1))
plt.title('GDP Forecast with 1 Sigma Error Range')
plt.show()

