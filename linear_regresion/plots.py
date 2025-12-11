# plot loss over epochs

import matplotlib.pyplot as plt

def plot_loss(loss_history):
    if not loss_history:
        print("No loss history to plot.")
        return

    plt.plot(loss_history)
    plt.title("MSE vs Epoch")
    plt.xlabel("epoch")
    plt.ylabel("mse")
    plt.grid(True)
    plt.show()
