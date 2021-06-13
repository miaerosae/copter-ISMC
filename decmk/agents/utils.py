import numpy as np


class CA():
    def __init__(self, B):
        self.B = B

    def get(self, fault_index):
        self.B[:, fault_index] = np.zeros((4, 1))
        Bf = self.B

        return Bf


class LoE:
    def __init__(self, time=0, index=0, level=1.0):
        self.time = time
        self.index = index
        self.level = level

    def get(self, t, u):
        effectiveness = self.get_effectiveness(t, u.shape[0])
        # effectiveness = np.ones_like(u)
        # if t >= self.time:
        #     effectiveness[self.index] = self.level
        return u * effectiveness.reshape(4, 1)  # , effectiveness

    def get_effectiveness(self, t, n):
        effectiveness = np.array([1] * n)
        if t >= self.time:
            effectiveness[self.index] = self.level
        return effectiveness


class FDI:
    def __init__(self, numact):
        self.state = np.eye(numact)

    def get_index(self, W):
        fault_index = np.where(np.diag(W) < 1)[0]
        return fault_index

    def get_W(self, u, uc):
        uc = np.clip(uc, 0, None)
        w = np.hstack([
            ui/uci if not np.isclose(uci, 0)
            else 1 if (np.isclose(ui, 0) and np.isclose(uci, 0))
            else 0
            for ui, uci in zip(u, uc)])
        self.state = np.diag(w)
        return np.diag(w)


if __name__ == "__main__":
    t = 4
    loe = LoE(time=3, index=0, level=0.5)
    loe1 = LoE(time=5, index=1, level=0)
    ca = CA(np.vstack(([1, 1, 1], [1, 1, 1], [1, 1, 1])))
    actuator_faults = [loe, loe1]
    print(ca.get(t, actuator_faults))
