import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

x = 10 * np.random.rand(50)
print(x)
y = 2 * x - 1 + np.random.randn(50)

model = LinearRegression(fit_intercept=True)
print(type(model))

print(x.shape)
print(y.shape)
print(x.ndim)
X = x[:,np.newaxis]
print(X.shape)
# print(X)
print(X.ndim)

model.fit(X,y)
print(model.coef_)
print(model.intercept_)
print(model.score(X,y))

xfit = np.linspace(-1,11,num=50)
Xfit = xfit[:,np.newaxis]
yfit = model.predict(Xfit)

plt.scatter(x,y)
plt.plot(xfit,yfit, color = 'r')
plt.show()