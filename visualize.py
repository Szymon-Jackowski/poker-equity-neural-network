import matplotlib.pyplot as plt
from train import train

if __name__ == "__main__":
    train_result = train("flop", 500, 300, 100, 0.1)    
    history_of_error=train_result[1]
    plt.plot(history_of_error)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Training error over epochs")
    plt.savefig("training_error.png")
    plt.show()