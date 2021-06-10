import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt

from fym.core import BaseEnv, BaseSystem
import fym.logging
from fym.utils.rot import angle2quat, quat2angle

from decmk.model.copter import Copter_nonlinear
from decmk.agents.ISMC import IntegralSMC_nonlinear
from decmk.agents.utils import CA, LoE
from copy import deepcopy


class Env(BaseEnv):
    def __init__(self):
        super().__init__(solver="odeint", max_t=20, dt=10, ode_step_len=100)
        self.plant = Copter_nonlinear()

        # Define faults
        self.actuator_faults = [
            LoE(time=3, index=0, level=0.0)
        ]

        # Define initial condition and reference at t=0
        ic = np.vstack((0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0))
        ref0 = np.vstack((0, -0, -0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0))

        # Define agents
        self.controller = IntegralSMC_nonlinear(self.plant.J,
                                                self.plant.m,
                                                self.plant.g,
                                                self.plant.d,
                                                ic,
                                                ref0)
        self.CA = CA(self.plant.mixer.B)

    def step(self):
        *_, done = self.update()
        return done

    def control_allocation(self, t, forces):
        Bf = self.CA.get(t, self.actuator_faults)
        rotors = Bf.dot(forces)

        return rotors

    def get_ref(self, t, x):
        # if t <= 5:
        #     pos_des = np.vstack((0, 0, -5))*10
        # elif 5 < t and t <= 10:
        #     pos_des = np.vstack((0, 0, -10))*10
        # elif 10 < t and t <= 15:
        #     pos_des = np.vstack((0, 0, -5))*10
        # else:
        #     pos_des = np.vstack((0, 0, 0))
        # pos_des = np.vstack((0, -0, -2))
        # vel_des = np.vstack((0, 0, 0))
        pos_des = np.vstack((cos(t), sin(t), -0.1*t))
        vel_des = np.vstack((-sin(t), cos(t), -0.1))
        quat_des = np.vstack((1, 0, 0, 0))
        omega_des = np.zeros((3, 1))
        ref = np.vstack((pos_des, vel_des, quat_des, omega_des))

        return ref

    def _get_derivs(self, t, x, p):
        ref = self.get_ref(t, x)

        forces = self.controller.get_FM(x, ref, p, t)

        # Controller
        rotors_cmd = self.control_allocation(t, forces)

        _rotors = np.clip(rotors_cmd, 0, self.plant.rotor_max)
        rotors = deepcopy(_rotors)

        return rotors_cmd, rotors, forces, ref

    def set_dot(self, t):
        x = self.plant.state
        p = self.controller.observe_list()

        rotors_cmd, rotors, forces, ref = self._get_derivs(t, x, p)

        self.plant.set_dot(t, rotors)
        self.controller.set_dot(x, ref)

    def logger_callback(self, i, t, y, *args):
        states = self.observe_dict(y)
        x_flat = self.plant.observe_vec(y[self.plant.flat_index])
        p = self.controller.observe_list(y[self.controller.flat_index])
        x = states["plant"]
        rotors_cmd, rotors, forces, ref = self._get_derivs(t, x_flat, p)

        return dict(t=t, x=x, rotors=rotors, rotors_cmd=rotors_cmd,
                    ref=ref)


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

    # Rotor
    plt.figure()

    ax = plt.subplot(411)
    plt.plot(data["t"], data["rotors"][:, 0], "k-", label="Response")
    plt.plot(data["t"], data["rotors_cmd"][:, 0], "r--", label="Command")
    plt.ylim([-5.1, 40])
    plt.legend()

    plt.subplot(412, sharex=ax)
    plt.plot(data["t"], data["rotors"][:, 1], "k-")
    plt.plot(data["t"], data["rotors_cmd"][:, 1], "r--")
    plt.ylim([-5.1, 40])

    plt.subplot(413, sharex=ax)
    plt.plot(data["t"], data["rotors"][:, 2], "k-")
    plt.plot(data["t"], data["rotors_cmd"][:, 2], "r--")
    plt.ylim([-5.1, 40])

    plt.subplot(414, sharex=ax)
    plt.plot(data["t"], data["rotors"][:, 3], "k-")
    plt.plot(data["t"], data["rotors_cmd"][:, 3], "r--")
    plt.ylim([-5.1, 40])

    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Rotor force")
    plt.tight_layout()

    plt.figure()

    ax = plt.subplot(411)
    plt.plot(data["t"], data["ref"][:, 0, 0], "r-", label="x (cmd)")
    plt.plot(data["t"], data["x"]["pos"][:, 0, 0], label="x")

    plt.plot(data["t"], data["ref"][:, 1, 0], "r--", label="y (cmd)")
    plt.plot(data["t"], data["x"]["pos"][:, 1, 0], "--", label="y")

    plt.plot(data["t"], data["ref"][:, 2, 0], "r-.", label="z (cmd)")
    plt.plot(data["t"], data["x"]["pos"][:, 2, 0], label="z")

    plt.subplot(412, sharex=ax)
    plt.plot(data["t"], data["x"]["vel"].squeeze())
    plt.legend([r'$u$', r'$v$', r'$w$'])
    plt.subplot(413, sharex=ax)
    plt.plot(data["t"], np.transpose(quat2angle(np.transpose(data["x"]["quat"].squeeze()))))
    plt.legend([r'$psi$', r'$theta$', r'$phi$'])
    plt.subplot(414, sharex=ax)
    plt.plot(data["t"], data["x"]["omega"].squeeze())
    plt.legend([r'$p$', r'$q$', r'$r$'])
    # plt.axvspan(3, 3.042, alpha=0.2, color="b")
    # plt.axvline(3.042, alpha=0.8, color="b", linewidth=0.5)

    # plt.axvspan(6, 6.011, alpha=0.2, color="b")
    # plt.axvline(6.011, alpha=0.8, color="b", linewidth=0.5)

    # plt.annotate("Rotor 0 fails", xy=(3, 0), xytext=(3.5, 0.5),
    #              arrowprops=dict(arrowstyle='->', lw=1.5))
    # plt.annotate("Rotor 2 fails", xy=(6, 0), xytext=(7.5, 0.2),
    #              arrowprops=dict(arrowstyle='->', lw=1.5))

    # plt.xlabel("Time, sec")
    # plt.ylabel("Position")
    # plt.legend(loc="right")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    exp1()
    exp1_plot()
