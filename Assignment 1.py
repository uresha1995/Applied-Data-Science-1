# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 22:40:58 2023

@author: uresha
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Data Set 1

#Read and print the CSV data file

dataset1 = pd.read_csv(r'C:\Users\uresha\Dropbox\PC\Desktop\UH\2. Applied Data Science 1\Assignment 1\WLD_RTFP_country_2023-10-02.csv')
print(dataset1)

#Check the Quality of the data set
#To get a summary of data set 

dataset1.info()
dataset1.describe()
    
#Return the number of missing values in dataset

dataset1.isnull().sum()

#Preparing data 

#Enhance the accuracy of the variables use for analysis
#Remove all the null records

dataset1 = dataset1.dropna(axis = 0)

dataset1['date'] = pd.to_datetime(dataset1['date'])
dataset1.set_index('date', inplace = True)

#Check the summary of data again to watch the accuracy

dataset1.info()
dataset1.describe()

#Graph: Line plot
#Inflation for all countries 
    
countries = dataset1['country'].unique()
  
plt.figure(figsize=(20, 12))

for country in countries:
    country_data = dataset1[dataset1['country'] == country]
    plt.plot(country_data['Inflation'], label = country)

plt.title('Inflation by Country')
plt.xlabel('Years')
plt.ylabel('Inflation Rate (%)')
plt.legend()
plt.grid(True)
plt.show()

#Graph 1: Line plot

#Inflation for selected countries 
    """ Since the above graph includes too many countreis, it's hard 
        to get an overall idea. Therefore, another line graph was 
        generated for selected countries of choice (Iraq, Afghanistan, 
        Myanmar, Nigeria, and South Sudan).
    """
    
#Plotting tha graph

sel_countries = ['Iraq', 'Afghanistan', 'Myanmar', 'Nigeria', 'South Sudan'] 

plt.figure(figsize=(10, 6))

for country in sel_countries:
    country_data = dataset1[dataset1['country'] == country]
    plt.plot(country_data['Inflation'], label=country)

plt.title('Inflation for Selected Countries')
plt.xlabel('Years')
plt.ylabel('Inflation Rate (%)')
plt.legend()
plt.grid(True)
plt.show()

#Data Set 2

#Read and print the CSV data file

dataset2 = pd.read_csv(r'C:\Users\uresha\Dropbox\PC\Desktop\UH\2. Applied Data Science 1\Assignment 1\Salary.csv')
print(dataset2)

#Check the Quality of the data set
#To get a summary of data set

dataset2.info()
dataset2.describe()

#Return the number of missing values in dataset

dataset2.isnull().sum()

#Graph 2: Stacked Histogram

#Plotting tha graph

male_salaries = dataset2[dataset2['Gender'] == 'Male']['Salary']
female_salaries = dataset2[dataset2['Gender'] == 'Female']['Salary']

plt.figure(figsize=(15, 10))

plt.hist([male_salaries, female_salaries], bins = 30, alpha = 0.7, 
         color = ['lightsalmon', 'forestgreen'], label = ['Male', 'Female'], 
         stacked = True)

plt.title('Salary Distribution by Gender')
plt.xlabel('Salary')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.show()

#Graph 3: Pie-Chart

#Plotting the graph

country = dataset2['Country'].value_counts()

plt.figure(figsize=(8, 8))

plt.pie(country, labels=country.index, autopct='%1.1f%%', startangle=140, 
        colors=['salmon', 'cornflowerblue', 'gold', 'orchid', 'lawngreen'])
plt.title('Country wise Distribution')
plt.axis('equal')  
plt.show()








