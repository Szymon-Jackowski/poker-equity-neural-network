# Poker Equity Neural Network

This project trains a small neural network, built from scratch with only NumPy, to predict poker hand equity (win probability) directly from cards, without running a Monte Carlo simulation at inference time. It's a companion project to my Kuhn Poker CFR solver: that one solves a simplified game exactly using game theory, this one approximates a much messier real-world quantity (Texas Hold'em equity) using a learned function.

## The problem

Given a hand and however many community cards are known (preflop, flop, turn, or river), what's the probability of winning against a random opponent hand? The exact answer requires enumerating or simulating all possible ways the remaining cards can come out. This project instead trains a neural network to learn that mapping directly from encoded cards to a single equity number between 0 and 1.

## Files

`hand_evaluator.py` scores any 7-card hand into a comparable rank (from high card to straight flush), including edge cases like the wheel straight (A-2-3-4-5).

`data_generator.py` runs Monte Carlo simulations to compute the true equity of a given hand, and generates labeled training data by sampling random hands at any stage of the game.

`neural_network.py` encodes cards as one-hot vectors (13 for rank, 4 for suit, per card) and implements a 2-layer network (ReLU hidden layer, sigmoid output) with manual forward and backward propagation.

`train.py` runs the training loop: for each stage of the game it builds a separate network (input size depends on how many cards are known), trains it on a random dataset, and can also track specific hands through training to see how their predictions evolve epoch by epoch.

`evaluate.py` measures mean squared error on a held-out test set generated independently from the training data.

`visualize.py` trains one network per stage (preflop, flop, turn, river) and saves all plots to `charts/`.

`charts/` contains all the generated plots: 4 training error curves and 8 hand convergence plots.

`requirements.txt` lists the two dependencies (numpy and matplotlib) needed to run this project.

## Why one network per stage

Preflop, flop, turn, and river all have a different number of known cards, so the input vector is a different size at each stage. Rather than padding everything to a common size, I trained four separate small networks, each sized for its stage (16 hidden neurons for preflop, up to 64 for river).

## Convergence on sample hands

For each stage I picked one strong and one weak example hand and tracked the network's prediction after every epoch, alongside the true equity from Monte Carlo:

- preflop: pocket aces (~0.85 equity) vs 7-2 offsuit (~0.35)
- flop: aces with top set on an A-K-7 board (~0.96) vs 7-2 missing entirely on A-K-T (~0.21)
- turn: kings full house (~1.00) vs a hand that missed the board entirely (~0.17)
- river: nut flush (~1.00) vs a weak hand with almost no help from the board (~0.10)

All predictions start around 0.5, which is expected: the weights start near zero, and sigmoid(0) is 0.5. From there they move quickly in the right direction. Weak hands converge to within a few points of the true equity line, typically by epoch 50 to 150. One of the weak hands usually shows a brief dip-and-overshoot before it settles (in this run it's weak_flop, dropping to about 0.17 before climbing back up); which hand does this varies between runs since the training data is resampled each time. Strong hands rise fast but flatten out just short of their target (for example strong_flop settles around 0.95 against a true value of 0.96).

This gap isn't noise, it comes from the sigmoid's own gradient. The backward pass includes a factor of output times (1 - output), which shrinks toward zero as the output approaches 0 or 1. Since strong hands have targets sitting right at that edge of the range, the gradient pushing them the rest of the way gets weaker and weaker, while weak hands sit closer to the middle where the gradient stays strong. On top of that, an equity of exactly 1.0 is mathematically out of reach for a sigmoid, and extreme hands like these are also rarer in a randomly sampled training set, so the network has less incentive to fit them perfectly.

## Training error by stage

The training error curves show a clear pattern: flop, turn, and river all converge to essentially zero error, meaning the network can nearly memorize its 500 training examples once enough cards are visible. Preflop plateaus at a noticeably higher error instead, because with only 2 known cards there's a lot of unavoidable variance in the outcome, no amount of fitting can remove randomness that depends on cards not yet dealt.

Starting error also increases with each stage (about 0.011 for preflop, up to 0.084 for river). That follows from how spread out the true equities are: on the river most hands are already decided one way or the other, so their equities sit near 0 or 1, far from the network's default 0.5 guess. Preflop equities cluster closer to 0.5 to begin with, so the initial error is smaller.

One side note from the data itself: Monte Carlo assigned the strong_turn example (kings full house) an equity of exactly 1.00, even though that hand isn't actually unbeatable at that point: an opponent already holding a pair of deuces would have quads right now, and a river card completing a different opponent holding into quads is also possible. With only 500 simulations, those specific opposing hands simply never got drawn.

## Test error

The training error curve for flop (see `charts/training_error_flop.png`) drops to near zero by the end of 200 epochs, but that number only reflects how well the network fits the 500 hands it was trained on. Running `evaluate.py` on 300 freshly generated flop hands, which the network never saw during training, gives a test MSE of about 0.048 (roughly 22 percentage points of equity off, on average). That gap between near-zero error on the training set and a much higher error on new hands is a sign of overfitting: with a training set this small relative to the number of possible flop hands, the network has enough capacity to fit it closely rather than learn a pattern general enough to hold up on hands it hasn't seen. A larger training set would likely narrow this gap.

## Parameters used

All results above were generated with 500 training examples, 500 Monte Carlo iterations per equity label, 200 training epochs, and a learning rate of 0.1. Since the training data is randomly sampled, re-running this will produce slightly different numbers, though the qualitative patterns described above should hold.

All the plots described above are saved in the `charts` folder: 4 training error curves (one per stage) and 8 convergence plots (one strong and one weak hand per stage).

## Running it

Install dependencies first: `pip3 install -r requirements.txt`

```
python3 train.py       # train a network for one stage
python3 evaluate.py    # check error on held-out test data
python3 visualize.py   # train all four stages and generate all plots
```