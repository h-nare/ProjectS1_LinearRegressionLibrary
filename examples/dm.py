from linear_regresion import LinearRegression

X = [
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 3],
]

y = [10, 12, 20, 22]   # Something linear

model = LinearRegression(learning_rate=0.01, epochs=1000)
model.fit(X, y)

print("Weights:", model.weights)
print("Bias:", model.bias)
print("MSE:", model.mse(X, y))

