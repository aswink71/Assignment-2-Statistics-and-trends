# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 02:49:58 2023

@author: aswin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, skew, kurtosis

def read_data_file(file_path):
    """
    Reads a data file in CSV format and returns a Pandas DataFrame.

    Parameters:
    file_path (str): The path to the data file.

    Returns:
    pandas.DataFrame: The data read from the file.
    """
    data = pd.read_csv(file_path, skiprows=4)
    data = data.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis=1)
    data = data.set_index('Country Name')
    return data

# read in population growth data
pop_data = read_data_file("C:\\Users\\aswin\\Desktop\\population growth.csv")
print("Population growth data description:")
print(pop_data.describe())

# calculate skewness and kurtosis for population growth data
pop_skewness = pop_data.skew()
pop_kurtosis = pop_data.kurtosis()

print("\nSkewness of population growth data:")
print(pop_skewness)
print("\nKurtosis of population growth data:")
print(pop_kurtosis)

# read in renewable energy consumption data
re_data = read_data_file("C:\\Users\\aswin\\Desktop\\renewable energy consumption.csv")
print("\nRenewable energy consumption data description:")
print(re_data.describe())

# calculate skewness and kurtosis for renewable energy consumption data
re_skewness = re_data.skew()
re_kurtosis = re_data.kurtosis()

print("\nSkewness of renewable energy consumption data:")
print(re_skewness)
print("\nKurtosis of renewable energy consumption data:")
print(re_kurtosis)

# select years to plot
years = ['1990', '1995', '2000', '2005', '2010', '2015', '2019']

# plot population growth for selected countries
pop_countries = ['Brazil', 'China', 'Finland', 'France', 'Germany', 'India', 'Indonesia', 'Italy', 'Japan', 'Mexico', 'Pakistan', 'United Kingdom', 'United States']
ax1 = pop_data.loc[pop_countries, years].T.plot(kind='bar', figsize=(10,6))
plt.title('Population growth (annual %)')
plt.xlabel('Year')
plt.ylabel('Population Growth (%)')
plt.ylim(0)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# plot renewable energy consumption for selected countries
re_countries = ['Brazil', 'China', 'Finland', 'France', 'Germany', 'India', 'Indonesia', 'Italy', 'Japan', 'Mexico', 'Pakistan', 'United Kingdom', 'United States']
ax2 = re_data.loc[re_countries, years].T.plot(kind='bar', figsize=(10,6))
plt.title('Renewable energy consumption (% of total final energy consumption)')
plt.xlabel('Year')
plt.ylabel('Renewable Energy Consumption (GWh)')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# calculate correlation between population growth and renewable energy consumption for selected countries
corr = pop_data.loc[pop_countries, years].corrwith(re_data.loc[re_countries, years])
print("\nCorrelation between population growth and renewable energy consumption for selected countries:")
print(corr)

# read in electric power consumption data
power_data = read_data_file("C:\\Users\\aswin\\Desktop\\electric power consumption.csv")
print("\nElectric power consumption data description:")
print(power_data.describe())

# calculate skewness and kurtosis for electric power consumption data
pop_skewness = pop_data.skew()
pop_kurtosis = pop_data.kurtosis()

print("\nSkewness of electric power consumption data:")
print(pop_skewness)
print("\nKurtosis of electric power consumption data:")
print(pop_kurtosis)

# read in co2 emission data
co2_data = read_data_file("C:\\Users\\aswin\\Desktop\\co2 emission.csv")
print("\nCO2 emission data description:")
print(co2_data.describe())

# calculate skewness and kurtosis for co2 emission data
pop_skewness = pop_data.skew()
pop_kurtosis = pop_data.kurtosis()

print("\nSkewness of co2 emission data:")
print(pop_skewness)
print("\nKurtosis of co2 emission data:")
print(pop_kurtosis)

# plot electric power consumption for selected countries
power_countries = ['United States', 'China', 'India', 'Brazil', 'Japan', 'Pakistan', 'Germany', 'Canada', 'France', 'United Kingdom', 'Italy']
power_data.loc[power_countries, '1990':'2020'].T.plot(figsize=(10,6))
plt.title('Electric power consumption (kWh per capita)')
plt.xlabel('Year')
plt.ylabel('Electric Power Consumption (GWh)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.gca().yaxis.grid(True)
plt.tight_layout()
plt.show()

# plot co2 emission for selected countries
co2_countries = ['United States', 'China', 'India', 'Brazil', 'Japan', 'Pakistan', 'Germany', 'Canada', 'France', 'United Kingdom', 'Italy']
co2_data.loc[co2_countries, '1990':'2020'].T.plot(figsize=(10,6))
plt.title('CO2 emissions (metric tons per capita)')
plt.xlabel('Year')
plt.ylabel('CO2 Emission (kt)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.gca().yaxis.grid(True)
plt.tight_layout()
plt.show()

# calculate correlation between electric power consumption and co2 emission for selected countries
corr = power_data.loc[power_countries, '1990':'2020'].corrwith(co2_data.loc[co2_countries, '1990':'2020'])
print("\nCorrelation between electric power consumption and co2 emission for selected countries:")
print(corr)

# create a DataFrame with data for China and the United States
data = {'Country': ['China', 'United States'], 
        'Electric power consumption (kWh per capita)': [7643, 9119], 
        'CO2 emissions (metric tons per capita)': [9.5, 13.0], 
        'Population growth (annual %)': [2.71, 1.75], 
        'Renewable energy consumption (% of total final energy consumption)': [5.8, 7.1]}
df = pd.DataFrame(data).set_index('Country')

# transpose the DataFrame
df_t = df.T

# plot heatmap
sns.heatmap(df_t, cmap='YlGnBu', annot=True, fmt='.1f', vmin=0, vmax=20)

plt.title('China and the United States Indicators Heatmap')
plt.show()
