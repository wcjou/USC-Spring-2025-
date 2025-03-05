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
# print(df.loc[(df['alcohol'].idxmax())])

#8
# random_wine = df.sample(1)
# print(random_wine['residual.sugar'].values[0])

#9
# qual_4 = df[df['quality'] == 4]
# print(qual_4.sample(1))

#10
# print(df.shape[0])
# df = df[df['quality'] != 4]
# print(df.shape[0])