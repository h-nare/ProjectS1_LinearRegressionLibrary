# test file for my linear regression model

from linear_regresion import LinearRegression, train_test_split
from linear_regresion.metrics import mse

# simple dataset with 2 features
X = [
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 3],
    [5, 5],
    [6, 7],
]

y = [10, 12, 20, 22, 30, 36]

# split into train/test
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.33)

# create model
model = LinearRegression(
    learning_rate=0.01,
    epochs=3000,
    normalize=True,
    early_stopping=True,
    patience=200,
    tol=1e-7,
    random_init=True
)

# train model
model.fit(X_train, y_train)

# show learned parameters
print("weights:", model.weights)
print("bias:", model.bias)

# predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# check accuracy
print("train mse:", mse(y_train, train_pred))
print("test mse:", mse(y_test, test_pred))

# plot training curve
model.plot_loss()

model.save("saved_model.json")
print("Model saved!")

# create a new model and load the saved parameters
loaded_model = LinearRegression()
loaded_model.load("saved_model.json")

print("Loaded weights:", loaded_model.weights)
print("Loaded bias:", loaded_model.bias)


