# Linear Regression Library

## Description

This project is a Python library that implements a linear regression model.  
It allows training a regression model on numerical data, evaluating its performance, and using it to make predictions.

The project is intended for educational purposes and experimentation, with a focus on understanding how linear regression models are trained and evaluated.

---

## Problem Addressed

The library solves the problem of fitting a linear regression model to a dataset and predicting output values based on input features.

It allows users to:
- Train a linear regression model with multiple features
- Minimize prediction error using gradient descent
- Evaluate model performance using mean squared error (MSE)
- Visualize training progress using a loss curve
- Save and load trained model parameters

---

## Features

- Multivariate linear regression  
- Gradient descent training  
- Feature normalization  
- Mean squared error (MSE) evaluation  
- Early stopping  
- Loss curve visualization  
- Saving and loading model parameters using JSON  

---

## Computational Complexity

Let:
- n be the number of data samples  
- d be the number of features  
- e be the number of training epochs  

The training process has an approximate time complexity of:

O(e × n × d)

This implementation is suitable for small to medium-sized datasets.

---

## How to Run the Project

From the project root directory, run:

```bash
python3 -m linear_regresion.demo

from linear_regresion import LinearRegression

model = LinearRegression(learning_rate=0.01, epochs=1000, normalize=True)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(predictions)

linear_regresion/
├── model.py    # Linear regression model implementation
├── metrics.py  # Evaluation metrics (MSE)
├── plots.py    # Loss curve visualization
├── utils.py    # Helper functions
├── demo.py     # Demonstration script
└── __init__.py
```
## Improvement Plan

Several improvements are planned for future versions of this library. One potential enhancement is the addition of regularization techniques such as L1 and L2 regularization to reduce overfitting. This would be implemented by modifying the loss function and gradient update rules.

Another planned feature is support for polynomial regression, which would allow the model to capture non-linear relationships by expanding input features. Mini-batch gradient descent could also be added to improve training efficiency on larger datasets.

Additional evaluation metrics beyond mean squared error, as well as automated unit tests, could further improve the robustness and usability of the library.



