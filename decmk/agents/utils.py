import numpy as np


class LoE:
    def __init__(self, time=0, index=0, level=1.0):
        self.time = time
        self.index = index
        self.level = level

    def get(self, t, u):
        effectiveness = self.get_effectiveness(t, u.shape[0])
        return u * effectiveness.reshape(4, 1)  # , effectiveness

    def get_effectiveness(self, t, effectiveness):
        if t >= self.time:
            effectiveness[self.index] = self.level
        return effectiveness


class FDI:
    def __init__(self, numact):
        self.state = np.eye(numact)

    def get_index(self, W):
        fault_index = np.where(np.diag(W) < 1)[0]
        return fault_index


if __name__ == "__main__":
    t = 4
    loe = LoE(time=3, index=0, level=0.5)
    loe1 = LoE(time=5, index=1, level=0)
    print(loe.get_effectiveness(t, 4))
    print(loe.get(t, np.vstack((1, 2, 3, 4))))
    W = np.diag((1, 0.5, 1, 1))
    fdi = FDI(4)
    print(fdi.get_index(W))
