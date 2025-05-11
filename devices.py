import numpy as np


class SimpleHMM:
    def __init__(self, A, B, pi):
        self.A = A  # Transition probabilities
        self.B = B  # Emission probabilities
        self.pi = pi  # Initial state probabilities
        self.n_states = A.shape[0]
        self.n_obs = B.shape[1]
    
    def sample(self, T):
        states = np.zeros(T, dtype=int)
        emissions = np.zeros(T, dtype=int)
        
        # Initial state
        states[0] = np.random.choice(self.n_states, p=self.pi)
        emissions[0] = np.random.choice(self.n_obs, p=self.B[states[0]])
        
        for t in range(1, T):
            states[t] = np.random.choice(self.n_states, p=self.A[states[t-1]])
            emissions[t] = np.random.choice(self.n_obs, p=self.B[states[t]])
        
        return states, emissions


def poisson_device(lam, T):
    # lam: scalar or array of shape (T,)
    
    if np.isscalar(lam):
        lam = np.full(T, lam)
    return np.random.poisson(lam)


class PoissonHMM:
    def __init__(self, A, lambdas, pi):
        self.A = A              # Transition matrix
        self.lambdas = lambdas  # Poisson rates per state
        self.pi = pi            # Initial state distribution
        self.n_states = len(pi)

    def sample(self, T):
        states = np.zeros(T, dtype=int)
        emissions = np.zeros(T, dtype=int)

        states[0] = np.random.choice(self.n_states, p=self.pi)
        emissions[0] = np.random.poisson(self.lambdas[states[0]])

        for t in range(1, T):
            states[t] = np.random.choice(self.n_states, p=self.A[states[t - 1]])
            emissions[t] = np.random.poisson(self.lambdas[states[t]])

        return states, emissions