# William Jou
# ITP 259 Spring 2025
# HW 2

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

winedf = pd.read_csv("USC-Spring-2025-/Spring 2025 Courses/ITP 259/Homework/HW2/wineQualityReds.csv")

winedf.drop('Wine', axis=1, inplace=True)

wine_quality = winedf['quality']

winedf.drop('quality', axis=1, inplace=True)

print(winedf)
print(wine_quality)

norm = Normalizer()
