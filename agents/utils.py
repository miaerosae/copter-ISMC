import numpy as np


class CA():
    '''
    Halim Alwi, Christopher Edwards, "Fault Tolerant conttol using sliding
    modes with on-line control allocation". Automatica, vol. 44, np. 7,
    pp. 1859-1866, 2008
    '''
    def __init__(self, B):
        self.B = B

    def get(self, t, actuator_faults):
        # set effectiveness matrix
        W = np.diag([1.0] * self.B.shape[1])
        n = len(actuator_faults)
        for i in range(n):
            if t < actuator_faults[i].time:
                W = W
            else:
                a = actuator_faults[i].index
                W[a, a] = actuator_faults[i].level
        # set pseudo-inverse
        Bf = np.linalg.pinv(self.B)
        for i in range(n):
            if t >= actuator_faults[i].time:
                Bf = W.dot(np.transpose(self.B))\
                    .dot(np.linalg.pinv(self.B.dot(W).dot(np.transpose(self.B))))

        return Bf


class LoE:
    def __init__(self, time=0, index=0, level=1.0):
        self.time = time
        self.index = index
        self.level = level


if __name__ == "__main__":
    t = 4
    loe = LoE(time=3, index=0, level=0.5)
    loe1 = LoE(time=5, index=1, level=0)
    ca = CA(np.vstack(([1, 1, 1], [1, 1, 1], [1, 1, 1])))
    actuator_faults = [loe, loe1]
    print(ca.get(t, actuator_faults))
