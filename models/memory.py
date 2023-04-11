import numpy as np

class ReplayBuffer():
    def __init__(self, max_size):
        self.storage = []  # empty list
        self.max_size = max_size
        self.pointer = 0

    def store_transition(self, transition):
        if len(self.storage) == self.max_size:  # replace the old data
            self.storage[self.pointer] = transition
            self.pointer = (self.pointer + 1) % self.max_size  # point to next position
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        # Define the array of indices for random sampling from storage
        # the size of this array equals to batch_size
        ind_array = np.random.randint(0, len(self.storage), size=batch_size)
        s, a, r, s_, d = [], [], [], [], []

        for i in ind_array:
            S, A, R, S_, D = self.storage[i]
            s.append(np.array(S, copy=False))
            a.append(np.array(A, copy=False))
            r.append(np.array(R, copy=False))
            s_.append(np.array(S_, copy=False))
            d.append(np.array(D, copy=False))
        return np.array(s), np.array(a), np.array(r).reshape(-1, 1), np.array(s_), np.array(d).reshape(-1, 1)