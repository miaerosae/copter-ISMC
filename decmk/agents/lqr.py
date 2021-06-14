import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lin

from decmk.agents.utils import quat2angle


def LQR(A: np.array, B: np.array, Q: np.array, R: np.array, with_eigs=False) \
        -> np.array:
    P = lin.solve_continuous_are(A, B, Q, R)
    if np.size(R) == 1:
        K = (np.transpose(B).dot(P)) / R
    else:
        K = np.linalg.inv(R).dot((np.transpose(B).dot(P)))

    eig_vals, eig_vecs = np.linalg.eig(A - B.dot(K))

    if with_eigs:
        return K, P, eig_vals, eig_vecs
    else:
        return K, P


class LQRController:
    def __init__(self, Jinv, m, g,
                 Q=np.diag([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]),
                 R=np.diag([1, 1, 1, 1]),
                 ):
        self.Jinv = Jinv
        self.m, self.g = m, g
        self.trim_forces = np.vstack([self.m * self.g, 0, 0, 0])

        '''
        linearized condition near trim pt
        state variables with {pos, vel, Euler angle, omega}
        '''
        A = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, -g, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        B = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [-1/m, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, Jinv[0, 0], 0, 0],
                      [0, 0, Jinv[1, 1], 0],
                      [0, 0, 0, Jinv[2, 2]]])

        self.K, *_ = LQR(A, B, Q, R)

    def transform(self, y):
        # input comes in a form {pos, vel, quat, omega}
        if len(y) == 13:
            return np.vstack((y[0:6],
                              np.vstack(quat2angle(y[6:10])[::-1]), y[10:]))

    def get_FM(self, obs, ref):
        x = self.transform(obs)
        x_ref = self.transform(ref)
        forces = -self.K.dot(x - x_ref) + self.trim_forces
        return forces


if __name__ == "__main__":
    pass
