# simple train-test split

import random

def train_test_split(X, y, test_size=0.2, shuffle=True):
    data = list(zip(X, y))

    if shuffle:
        random.shuffle(data)  # mix pairs

    split_index = int(len(data) * (1 - test_size))

    train = data[:split_index]
    test = data[split_index:]

    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    return list(X_train), list(y_train), list(X_test), list(y_test)
