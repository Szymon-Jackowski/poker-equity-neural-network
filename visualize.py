import matplotlib.pyplot as plt
import os
from train import train

EXAMPLES = {
    "preflop": {
        "strong_preflop": [(14, "Spades"), (14, "Hearts")],
        "weak_preflop":   [(7, "Hearts"), (2, "Diamonds")],
    },
    "flop": {
        "strong_flop": [(14, "Spades"), (14, "Hearts"), (14, "Diamonds"), (13, "Clubs"), (7, "Hearts")],
        "weak_flop":   [(7, "Hearts"), (2, "Diamonds"), (14, "Spades"), (13, "Diamonds"), (10, "Clubs")],
    },
    "turn": {
        "strong_turn": [(13, "Hearts"), (13, "Diamonds"), (13, "Spades"), (7, "Clubs"), (2, "Diamonds"), (2, "Hearts")],
        "weak_turn":   [(5, "Clubs"), (3, "Diamonds"), (14, "Spades"), (13, "Hearts"), (12, "Diamonds"), (11, "Clubs")],
    },
    "river": {
        "strong_river": [(14, "Spades"), (12, "Spades"), (2, "Spades"), (7, "Spades"), (9, "Spades"), (13, "Diamonds"), (4, "Clubs")],
        "weak_river":   [(7, "Hearts"), (2, "Diamonds"), (14, "Spades"), (13, "Diamonds"), (12, "Clubs"), (11, "Hearts"), (4, "Spades")],
    },
}

if __name__=="__main__":
    os.makedirs("charts", exist_ok=True)
    for stage, stage_examples in EXAMPLES.items():
        train_result = train(stage, 500, 500, 200, 0.1, list(stage_examples.values()))
        history_of_error=train_result[1]
        hands_equity=train_result[2]
        plt.figure()
        plt.plot(history_of_error)
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.title(f"Training error over epochs ({stage})")
        plt.savefig(f"charts/training_error_{stage}.png")
        plt.close()
        for name, cards in stage_examples.items():
            values = hands_equity[tuple(cards)]
            true_equity = values[0]
            predictions = [p.item() for p in values[1:]]
            plt.figure()
            plt.plot(predictions, label="network prediction")
            plt.axhline(true_equity, color="red", linestyle="--", label=f"Monte Carlo equity = {true_equity:.2f}")
            plt.ylim(0, 1)
            plt.xlabel("Epoch")
            plt.ylabel("Equity")
            plt.title(name)
            plt.legend()
            plt.savefig(f"charts/{name}.png")
            plt.close()