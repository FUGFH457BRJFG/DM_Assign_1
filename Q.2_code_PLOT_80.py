import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv('/Users/shristisingh/Philadelphia_Crime_Rate_noNA.csv')  


# View the first few rows of the dataset
print(data.head())

# Assume 'Price' is the target variable and the rest are features
# Adjust the column names based on your dataset
X = data.drop(columns=['HousePrice','Name','County'])  # Features
y = data['HousePrice']  # Target


# Define the data
crime_rate = data['CrimeRate']
miles_phila = data['MilesPhila']
pop_chg = data['PopChg']
house_price = data['HousePrice']

# Determine x-axis limits and ticks based on a fixed interval
x_min = min(crime_rate.min(), miles_phila.min(), pop_chg.min())
x_max = 90
x_ticks = np.arange(0, x_max + 10, 10)  # Adjust the step as needed

# Create the subplots
plt.figure(figsize=(12, 8))

# CrimeRate vs HousePrice
plt.subplot(3, 1, 1)
plt.scatter(crime_rate, house_price)
plt.xlabel('CrimeRate')
plt.ylabel('HousePrice')
plt.title('CrimeRate vs HousePrice')
plt.xticks(x_ticks)  # Set x-axis ticks
plt.xlim(x_min, x_max)

# MilesPhila vs HousePrice
plt.subplot(3, 1, 2)
plt.scatter(miles_phila, house_price)
plt.xlabel('MilesPhila')
plt.ylabel('HousePrice')
plt.title('MilesPhila vs HousePrice')
plt.xticks(x_ticks)  # Set x-axis ticks
plt.xlim(x_min, x_max)

# PopChg vs HousePrice
plt.subplot(3, 1, 3)
plt.scatter(pop_chg, house_price)
plt.xlabel('PopChg')
plt.ylabel('HousePrice')
plt.title('PopChg vs HousePrice')
plt.xticks(x_ticks)  # Set x-axis ticks
plt.xlim(x_min, x_max)

# Adjust layout
plt.tight_layout()
plt.show()

 