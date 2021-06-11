import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt

from fym.core import BaseEnv, BaseSystem
import fym.logging
from fym.utils.rot import angle2quat, quat2angle

from decmk.model.copter import Copter_nonlinear
from decmk.agents.lqr import LQRController
from decmk.agents.utils import CA, LoE, FDI
from copy import deepcopy


class Env(BaseEnv):
    def __init__(self):
        super().__init__(max_t=10, dt=0.01)

        # Define faults
        self.actuator_faults = [
            # LoE(time=0, index=0, level=0.0),
            # LoE(time=14, index=3, level=0.3)
        ]

        # Define agents
        self.plant = Copter_nonlinear()
        n = self.plant.mixer.B.shape[1]
        self.fdi = FDI(numact=n)
        self.controller = LQRController(self.plant.Jinv,
                                        self.plant.m,
                                        self.plant.g)
        self.CA = CA(self.plant.mixer.B)

    def step(self):
        *_, done = self.update()
        return done

    def get_ref(self, t, x):
        # if t <= 5:
        #     pos_des = np.vstack((0, 0, -5))*10
        # elif 5 < t and t <= 10:
        #     pos_des = np.vstack((0, 0, -10))*10
        # elif 10 < t and t <= 15:
        #     pos_des = np.vstack((0, 0, -5))*10
        # else:
        #     pos_des = np.vstack((0, 0, 0))
        pos_des = np.vstack((0, -0, -10))
        vel_des = np.vstack((0, 0, 0))
        # pos_des = np.vstack((cos(t), sin(t), -1*t))
        # vel_des = np.vstack((-sin(t), cos(t), -1))
        quat_des = np.vstack((1, 0, 0, 0))
        omega_des = np.zeros((3, 1))
        ref = np.vstack((pos_des, vel_des, quat_des, omega_des))

        return ref

    def control_allocation(self, t, forces, W):
        fault_index = self.fdi.get_index(W)

        if len(fault_index) == 0:
            rotors = np.linalg.pinv(self.plant.mixer.B.dot(W)).dot(forces)
        else:
            Bf = self.CA.get(fault_index)
            rotors = np.linalg.pinv(Bf.dot(W)).dot(forces)

        return rotors

    def _get_derivs(self, t, x):
        ref = self.get_ref(t, x)
        W = self.fdi.state
        fault_index = self.fdi.get_index(W)

        forces = self.controller.get_FM(x, ref)

        # Controller
        rotors_cmd = self.control_allocation(t, forces, W)

        _rotors = np.clip(rotors_cmd, 0, self.plant.rotor_max)
        rotors = deepcopy(_rotors)

        for act_fault in self.actuator_faults:
            rotors = act_fault.get(t, rotors)

        _rotors[fault_index] = 1
        W = self.fdi.get_W(rotors, _rotors)

        return rotors_cmd, rotors, forces, ref

    def set_dot(self, t):
        x = self.plant.state
        rotors_cmd, rotors, forces, ref = self._get_derivs(t, x)

        self.plant.set_dot(t, rotors)

    def logger_callback(self, i, t, y, *args):
        x = self.plant.state
        rotors_cmd, rotors, forces, ref = self._get_derivs(t, x)
        return dict(t=t, **self.observe_dict(), forces=forces, rotors=rotors,
                    rotors_cmd=rotors_cmd, ref=ref)


def run():
    env = Env()
    env.logger = fym.logging.Logger("data.h5")

    env.reset()

    while True:
        env.render()
        done = env.step()

        if done:
            break

    env.close()


def exp1():
    run()


def exp1_plot():
    data = fym.logging.load("data.h5")

    fig = plt.figure()
    fig.suptitle("[LQR] state variables")
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
    ax4 = fig.add_subplot(4, 1, 4, sharex=ax1)

    ax1.plot(data['t'], data['plant']['pos'].squeeze(), label="plant")
    ax1.plot(data["t"], data["ref"][:, 0, 0], "r--", label="x (cmd)")
    ax1.plot(data["t"], data["ref"][:, 1, 0], "r--", label="y (cmd)")
    ax1.plot(data["t"], data["ref"][:, 2, 0], "r--", label="z (cmd)")
    ax2.plot(data['t'], data['plant']['vel'].squeeze())
    ax3.plot(data['t'], data['plant']['quat'].squeeze())
    ax4.plot(data['t'], data['plant']['omega'].squeeze())

    ax1.set_ylabel('Position [m]')
    ax1.legend([r'$x$', r'$y$', r'$z$'])
    ax1.grid(True)

    ax2.set_ylabel('Velocity [m/s]')
    ax2.legend([r'$u$', r'$v$', r'$w$'])
    ax2.grid(True)

    ax3.set_ylabel('Quaternian')
    ax3.legend([r'$p0$', r'$p1$', r'$p2$', r'$p3$'])
    ax3.grid(True)

    ax4.set_ylabel('Omega [rad/s]')
    ax4.legend([r'$p$', r'$q$', r'$r$'])
    ax4.set_xlabel('Time [sec]')
    ax4.grid(True)

    plt.tight_layout()

    fig2 = plt.figure()
    fig2.suptitle("[LQR] rotor input")
    ax1 = fig2.add_subplot(4, 1, 1)
    ax2 = fig2.add_subplot(4, 1, 2, sharex=ax1)
    ax3 = fig2.add_subplot(4, 1, 3, sharex=ax1)
    ax4 = fig2.add_subplot(4, 1, 4, sharex=ax1)

    ax1.plot(data['t'], data['rotors'].squeeze()[:, 0])
    ax2.plot(data['t'], data['rotors'].squeeze()[:, 1])
    ax3.plot(data['t'], data['rotors'].squeeze()[:, 2])
    ax4.plot(data['t'], data['rotors'].squeeze()[:, 3])

    ax1.set_ylabel('rotor1')
    ax1.grid(True)
    ax2.set_ylabel('rotor2')
    ax2.grid(True)
    ax3.set_ylabel('rotor3')
    ax3.grid(True)
    ax4.set_ylabel('rotor4')
    ax4.grid(True)
    ax4.set_xlabel('Time [sec]')

    plt.tight_layout()

    # fig3 = plt.figure()
    # fig3.suptitle("[LQR] force and moment input")
    # ax1 = fig3.add_subplot(4, 1, 1)
    # ax2 = fig3.add_subplot(4, 1, 2, sharex=ax1)
    # ax3 = fig3.add_subplot(4, 1, 3, sharex=ax1)
    # ax4 = fig3.add_subplot(4, 1, 4, sharex=ax1)

    # ax1.plot(data['t'], data['forces'].squeeze()[:, 0])
    # ax2.plot(data['t'], data['forces'].squeeze()[:, 1])
    # ax3.plot(data['t'], data['forces'].squeeze()[:, 2])
    # ax4.plot(data['t'], data['forces'].squeeze()[:, 3])

    # ax1.set_ylabel('F')
    # ax1.grid(True)
    # ax2.set_ylabel('M1')
    # ax2.grid(True)
    # ax3.set_ylabel('M2')
    # ax3.grid(True)
    # ax4.set_ylabel('M3')
    # ax4.grid(True)
    # ax4.set_xlabel('Time [sec]')

    plt.show()


if __name__ == "__main__":
    exp1()
    exp1_plot()
