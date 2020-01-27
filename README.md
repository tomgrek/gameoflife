# Learnable Game Of Life (John Conway) with convolutions, in PyTorch

Learnable cellular automata... in PyTorch, using convolutions.

Just a fun thing to try. Documented [here](https://medium.com/@tomgrek/evolving-game-of-life-neural-networks-chaos-and-complexity-94b509bc7aa8).

## Instructions

1. Run `python3 life.py` just to see the basic game in action. Press `q` to exit.

![A basic game run according to the rules](https://user-images.githubusercontent.com/2245347/73218271-7f575980-410e-11ea-9807-dc25a222e5e3.gif)

2. Run `python3 lifedatagen.py` to start collecting some training data. Press `q` to exit once it gets fairly stable/boring.
2a. Edit the code of `lifedatagen.py` at change line 21 from `if False:` to `if True:`. The first time you run the script,
it creates a new file (a numpy vector). The second time it'll append to that vector instead (provided you make the change.)
2b. Run `python3 lifedatagen.py` several more times to gather training examples.
2c. You should see a file called `train.data` has been created.

3. Run `python3 lifeshuffle.py`. This takes `train.data`, shuffles it, and removes discontinuities from when you exited/restarted in step 2b above.

4. Run `python3 lifetrain.py`. This trains the network. Once training is done (GPU required, likely only 5-10 minutes) it'll run the modelled game and display it. Press `q` to quit or `r` to restart with a new randomly initialized board. Some params like learning rate, num epochs can be edited inside that file.

Here's an example output:

![A learned game](https://user-images.githubusercontent.com/2245347/73218288-8e3e0c00-410e-11ea-8bdc-49eef2b91d8e.gif)