import pandas as pd
#1 
df = pd.read_csv('USC-Spring-2025-\Spring 2025 Courses\ITP 259\EXAM_1_Guide\wineQualityReds.csv')

#2
# print(df.head(10))

#3
# print(df.sort_values(by='volatile.acidity', ascending=False))

#4
# print(df[df['quality'] == 7])

#5
# print(df['pH'].mean())

#6
# print(df[df['alcohol'] > 10].shape[0])

#7
print(df.loc[(df['alcohol'].idxmax())])