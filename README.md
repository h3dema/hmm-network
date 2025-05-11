# Simulation scenarios with HMM

A **Hidden Markov Model (HMM)** is a statistical model used to represent systems that transition between different states, where the states themselves are not directly observable.
Instead, you can infer them based on observable outputs.
HMMs are widely used in fields like speech recognition, bioinformatics, and signal processing.

## Test to generate simulated traffic

- `Simple HMM.ipynb`
- `Simple HMM-Poisson.ipynb`
- `Simple Poisson HMM+HMM.ipynb`
- `HMM Collision.ipynb`

## Predicting collisions

### Assume that we know the number of devices that are transmitting

- `Collision Likelihood with Random Forest.ipynb`
- `Collision Likelihood with transformer.ipynb`
- `Collision Likelihood with transformer (improved).ipynb`: better version of `Collision Likelihood with transformer.ipynb`. Uses extra information.
- `Collision Likelihood with MLP.ipynb`: same as before but uses MLP

### Don`t assume we know the transmitting devices

- `Collision Likelihood with transformer without emissions.ipynb`: uses proxy information to predict collisions
- `Collision Likelihood with TCNN without emissions.ipynb`: same as before changing the model to a Temporal CNN