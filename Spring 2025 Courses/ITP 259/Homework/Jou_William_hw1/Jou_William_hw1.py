# William Jou
# ITP 259 Spring 2025
# HW 1

import pandas as pd

# # 1.	Read the dataset into a dataframe. Be sure to import the header.
df = pd.read_csv('Homework\Jou_William_hw1\wineQualityReds.csv')
print(df.head())


# 2.	Print the first 10 rows of the dataframe.
print(df.head(10))


# 3.	Print the dataframe in descending order of volatility.
print(df.sort_values(by='volatile.acidity', ascending=False))


# 4.	Display those wines that have quality of 7.
print(df[df['quality'] == 7])


# 5.	What is the average pH of all wines?
print('Average pH:', df['pH'].mean())


# 6.	How many wines have alcohol level more than 10?
high_alcohol_df = df[df['alcohol'] > 10]
print(high_alcohol_df.shape[0], 'wines have an alcohol level more than 10')


# 7.	Which wine has the highest alcohol level?
print('Wine with Highest Alcohol:', df.loc[df['alcohol'].idxmax()])


# 8.	List the residual sugar level of a random wine.
random_wine = df.sample(1)
print('Random Wine Residual Sugar:', random_wine['residual.sugar'].values[0])


# 9.	List a random wine that has quality of 4.
df_low_quality = df[df['quality'] == 4]
print('Random Wine of Quality 4:', df_low_quality.sample(1))


# 10.	Drop wines that have quality 4. How many wines are left in the dataframe?
new_df = df[df['quality'] != 4]
print('Wines without a quality of 4:', new_df.shape[0])