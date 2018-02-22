## Computation of optimal transport and related hedging problems via neural networks and penalization
____________________________

This repository contains the implementations of the numerical examples in ["Computation of optimal transport and related hedging problems via neural networks and penalization"](https://arxiv.org/) (Link to be inserted)

### Prerequisites

- Python, Tensorflow, NumPy, Matplotlib, Seaborn, Pandas
- The programs were tested with Python 3.5.3

### Running the programs

- Each folder corresponds to one subsection in chapter 4 of the paper.
- Each program is self-contained and the default parameters are as used for the paper. At the bottom of each file, parameter values can be adjusted.
- Some output text files required for the plot programs are included. 

### GAN Toy
This folder contains slightly adjusted code from https://github.com/igul222/improved_wgan_training.
The original license is included in the folder. We thank the original author for making this code available!

The only things changed in the programs are
- adjusting parts of the code to Python 3,
- changing the implementation of the toy models to the one described in our paper (i.e. mainly changing the penalization term), and allowing for different types of transport-type GANs, not just the Wasserstein-1 GAN.

### License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details