import pandas as pd

#1 
df = pd.read_csv('USC-Spring-2025-\Spring 2025 Courses\ITP 259\EXAM_1_Guide\wineQualityReds.csv')

#2
df.drop('Wine', axis=1, inplace=True)
print(df)