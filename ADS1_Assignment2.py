# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:50:27 2023

@author: uresha
"""
# ADS1 - Assignment 2

"""
This assignment explores the connection between CO2 production and GDP. 
Throughout the report, the inter-relationships between the relevant variables, 
the nature of the annual data changes, and how they affected GDP country-wise
and year-wise from 2015 to 2020 evaluate.
"""

# Read the data




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
import seaborn as sns
def read_data():
    """
    Read and return the data File
    """
    original_dataset = pd.read_csv(
        r'C:\Users\uresha\Dropbox\PC\Desktop\UH\2. Applied Data Science 1\Assignment 2\DataSet1.csv')
    return original_dataset


dataset = read_data()
print(dataset)

# Read and transpose the data


def transpose_and_clean_dataset(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Preparing dataset
    columns_to_remove = ['Series Code', 'Country Code']
    df = df.drop(columns=columns_to_remove)

    # Transpose the years to a 'Year' column
    df = df.melt(id_vars=['Series Name', 'Country Name'],
                 var_name='Year', value_name='Value')

    # Convert the 'Year' column to a numeric type
    df['Year'] = pd.to_numeric(df['Year'].str.extract(r'(\d+)')[0])

    # Set 'Year' as the new index
    df = df.set_index('Year')

    return df


# Actual CSV file
result_df = transpose_and_clean_dataset(
    r'C:\Users\uresha\Dropbox\PC\Desktop\UH\2. Applied Data Science 1\Assignment 2\DataSet1.csv')

# Print DataFrame
print(result_df)

# Save the transposed DataFrame to a new CSV file
result_df.to_csv(
    r'C:\Users\uresha\Dropbox\PC\Desktop\UH\2. Applied Data Science 1\Assignment 2\transposed_dataset.csv')

# Explore the statistical properties
countries = dataset['Country Name'].unique()

# .describe()
for country in countries:
    country_data = dataset[dataset['Country Name'] == country]

    # Print summary statistics
    print(f'Summary Statistics for {country}:\n{country_data.describe()}')

# Two other statistical tools
# Skewness


def skew(dist):
    """
    Calculates the centralised and normalised skewness.
    """

    # calculates average and std, dev for centralising and normalising
    aver = np.mean(dist)
    std = np.std(dist)

    # now calculate the skewness
    value = np.sum(((dist-aver) / std)**3) / (len(dist) - 1)

    return value


dist = pd.DataFrame(result_df)

# Call the function
skewness_value = skew(result_df['Value'])
rounded_skewness_value = np.round(skewness_value, 3)
print(f"Skewness: {rounded_skewness_value}")

# Kutosis


def kurtosis(dist):
    """
    Calculates the centralised and normalised kurtosis.
    """

    # calculates average and std, dev for centralising and normalising
    aver = np.mean(dist)
    std = np.std(dist)

    # now calculate the kurtosis
    value = np.sum(((dist-aver) / std)**4) / len(dist-3) - 3.0

    return value


dist = pd.DataFrame(result_df)

# Call the function
kurtosis_value = kurtosis(result_df['Value'])
rounded_kurtosis_value = np.round(kurtosis_value, 3)
print(f"Kurtosis: {rounded_kurtosis_value}")

# 1st Graph


def barplot_co2_emission(result_df, series_name='CO2 emissions (kt)', figsize=(12, 6)):
    """
    Multiple bar graph to evaluate the CO2 emission of countries, year wise. 
    """
    # Filter data for Co2 emission
    co2_data = result_df[result_df['Series Name'] == series_name]

    # Pivot the DataFrame
    pivot_data = co2_data.pivot_table(
        index='Country Name', columns='Year', values='Value')

    # Plotting
    pivot_data.plot(kind='bar', stacked=False, figsize=(10, 6))

    # Add labels and title
    plt.xlabel('Countries')
    plt.ylabel('CO2 Emission (kt)')
    plt.title('CO2 Emission by Country and Year')

    # Show the plot
    plt.show()


# Call the function
barplot_co2_emission(result_df)

# 2nd Graph


def barplot_GDP_per_capita(result_df, series_name='GDP (current US$)', figsize=(12, 6), colormap='Spectral'):
    """
    Multiple bar graph to evaluate the GDP (current US$), year wise. 
    """
    # Filter data for GDP per capita (current US$)
    GDP_per_capita_data = result_df[result_df['Series Name'] == series_name]

    # Pivot the DataFrame
    pivot_data = GDP_per_capita_data.pivot_table(
        index='Country Name', columns='Year', values='Value')

    # Plotting
    pivot_data.plot(kind='bar', stacked=False,
                    figsize=(10, 6), colormap=colormap)

    # Add labels and title
    plt.xlabel('Countries')
    plt.ylabel('Current US$')
    plt.title('GDP (US$) by Country and Year')

    # Show the plot
    plt.show()


# Call the function
barplot_GDP_per_capita(result_df, colormap='Spectral')

# 3rd Graph
"""
    Heatmap to identify and measure the relationship between 
    the variables in China. 
    """

# Filter data for the China
china_data = result_df[result_df['Country Name'] == 'China']

# Pivot the DataFrame
pivot_data = china_data.pivot_table(
    index='Year', columns='Series Name', values='Value')

# Compute the correlation matrix using the DataFrame statistical method corr()
correlation_matrix = pivot_data.corr()

# Create a heatmap using seaborn
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True,
            cmap='mako', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap for China')
plt.show()

# 4th Graph
"""
    Heatmap to identify and measure the relationship between 
    the variables in United States. 
    """

# Filter data for the China
us_data = result_df[result_df['Country Name'] == 'United States']

# Pivot the DataFrame for easy correlation analysis
pivot_data = us_data.pivot_table(
    index='Year', columns='Series Name', values='Value')

# Compute the correlation matrix using the DataFrame statistical method corr()
correlation_matrix = pivot_data.corr()

# Create a heatmap using seaborn
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True,
            cmap='flare', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap for United States')
plt.show()

# 5th Graph


def linegraph_GDP_per_capita_growth(result_df, series_name='GDP per capita growth (annual %)', figsize=(12, 6), colormap='Spectral'):
    """
    Line graph to visualize the GDP per capita growth (annual %) 
    of countries over the years.
    """
    # Filter the GDP per capita growth series
    GDP_per_capita_growth_data = result_df[result_df['Series Name'] == series_name]

    # Pivot the DataFrame
    pivot_data = GDP_per_capita_growth_data.pivot_table(
        index='Country Name', columns='Year', values='Value')

    # Plotting
    plt.figure(figsize=figsize)
    for country in pivot_data.index:
        plt.plot(pivot_data.columns,
                 pivot_data.loc[country], label=country, linewidth=2)

    # Add labels and title
    plt.xlabel('Year')
    plt.title(f'{series_name} by Country and Year')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Show the plot
    plt.show()


# Call the function
linegraph_GDP_per_capita_growth(result_df, colormap='Spectral')

# 6th Graph


def linegraph_EIL_of_primary_energy(result_df, series_name='Energy intensity level of primary energy (MJ/$2017 PPP GDP)', figsize=(12, 6), colormap='Spectral'):
    """
    Line graph to visualize the GDP per capita growth (annual %) of countries over the years.
    """
    # Filter the GDP per capita growth series
    EIL_of_primary_energy_data = result_df[result_df['Series Name'] == series_name]

    # Pivot the DataFrame
    pivot_data = EIL_of_primary_energy_data.pivot_table(
        index='Country Name', columns='Year', values='Value')

    # Plotting
    plt.figure(figsize=figsize)
    for country in pivot_data.index:
        plt.plot(pivot_data.columns,
                 pivot_data.loc[country], label=country, linewidth=2, linestyle='--')

    # Add labels and title
    plt.xlabel('Year')
    plt.title(f'{series_name} by Country and Year')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Show the plot
    plt.show()


# Call the function
linegraph_EIL_of_primary_energy(result_df, colormap='Spectral')

# 7th table


"""
    Table to visualize the CO2 emissions (metric tons per capita) of 
    countries over the years.
    """

# Filter data for CO2 emissions series
co2_emi_data = result_df[result_df['Series Name']
                         == 'CO2 emissions (metric tons per capita)']

# Group by 'Country Name' and pivot the DataFrame
pivot_data = co2_emi_data.groupby(['Country Name', 'Year'])[
    'Value'].mean().unstack()

# Display the table using matplotlib
plt.figure(figsize=(5, 3))
# Turn off axis labels and ticks for good appearance
plt.axis('off')
plt.title('CO2 Emissions (metric tons per capita) by Country and Year')

# Display the table
table = plt.table(cellText=pivot_data.values, rowLabels=pivot_data.index,
                  colLabels=pivot_data.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.auto_set_column_width(col=list(range(len(pivot_data.columns))))

plt.show()

# 8th Graph


"""
    Heatmap to identify and measure the relationship between the variables 
    in India. 
    """

# Filter data for the India
india_data = result_df[result_df['Country Name'] == 'India']

# Pivot the DataFrame for easy correlation analysis
pivot_data = india_data.pivot_table(
    index='Year', columns='Series Name', values='Value')

# Compute the correlation matrix using the DataFrame statistical method corr()
correlation_matrix = pivot_data.corr()

# Create a heatmap using seaborn
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True,
            cmap='crest', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap for India')
plt.show()
