import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt

from fym.core import BaseEnv, BaseSystem
import fym.logging
from fym.utils.rot import angle2quat, quat2angle

from decmk.model.copter import Copter_nonlinear
from decmk.agents.AdaptiveISMC import AdaptiveISMC_nonlinear
from decmk.agents.utils import CA, LoE, FDI
from copy import deepcopy


class Env(BaseEnv):
    def __init__(self):
        super().__init__(solver="odeint", max_t=10, dt=5, ode_step_len=100)

        # Define faults
        self.actuator_faults = [
            LoE(time=5, index=0, level=0.0),
            # LoE(time=10, index=3, level=0.3)
        ]

        # Define initial condition and reference at t=0
        ic = np.vstack((0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0))
        ref0 = np.vstack((0, 0, -2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0))

        # Define agents
        self.plant = Copter_nonlinear()
        self.n = self.plant.mixer.B.shape[1]
        self.fdi = FDI(numact=self.n)
        self.controller = AdaptiveISMC_nonlinear(self.plant.J,
                                                 self.plant.m,
                                                 self.plant.g,
                                                 self.plant.d,
                                                 ic,
                                                 ref0)
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
        pos_des = np.vstack((0, 0, -2))
        vel_des = np.vstack((0, 0, 0))
        # pos_des = np.vstack((cos(t), sin(t), -t))
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

        d = self.plant.d
        c = self.plant.c
        self.plant.mixer.B = np.array([[1, 1, 1, 1],
                           [0, -d, 0, d],
                           [d, 0, -d, 0],
                           [-c, c, -c, c]])
        return rotors

    def _get_derivs(self, t, x, p, gamma, W):
        ref = self.get_ref(t, x)
        # fault_index = self.fdi.get_index(W)

        forces, sliding = self.controller.get_FM(x, ref, p, gamma, t)

        # Controller
        rotors_cmd = self.control_allocation(t, forces, W)

        _rotors = np.clip(rotors_cmd, 0, self.plant.rotor_max)
        rotors = deepcopy(_rotors)

        for act_fault in self.actuator_faults:
            rotors = act_fault.get(t, rotors)
            # effectiveness = act_fault.get_effectiveness(t, self.n)

        # W = np.diag(effectiveness)
        # _rotors[fault_index] = 1
        # W = self.fdi.get_W(rotors, _rotors)

        return rotors_cmd, rotors, forces, ref, sliding

    def set_dot(self, t):
        x = self.plant.state
        p, gamma = self.controller.observe_list()
        for act_fault in self.actuator_faults:
            effectiveness = act_fault.get_effectiveness(t, self.n)

        W = np.diag(effectiveness)

        rotors_cmd, rotors, forces, ref, sliding = \
            self._get_derivs(t, x, p, gamma, W)

        self.plant.set_dot(t, rotors)
        self.controller.set_dot(x, ref, sliding)

    def logger_callback(self, i, t, y, *args):
        states = self.observe_dict(y)
        x_flat = self.plant.observe_vec(y[self.plant.flat_index])
        ctrl_flat = self.controller.observe_list(y[self.controller.flat_index])
        x = states["plant"]
        for act_fault in self.actuator_faults:
            effectiveness = act_fault.get_effectiveness(t, self.n)

        W = np.diag(effectiveness)
        rotors_cmd, rotors, forces, ref, sliding = \
            self._get_derivs(t, x_flat, ctrl_flat[0], ctrl_flat[1], W)

        return dict(t=t, x=x, rotors=rotors, rotors_cmd=rotors_cmd,
                    ref=ref, gamma=ctrl_flat[1], forces=forces)


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

    ax = plt.subplot(611)
    plt.plot(data["t"], data["rotors"][:, 0], "k-", label="Response")
    plt.plot(data["t"], data["rotors_cmd"][:, 0], "r--", label="Command")
    plt.ylim([-5.1, 40])
    plt.legend()

    plt.subplot(612, sharex=ax)
    plt.plot(data["t"], data["rotors"][:, 1], "k-")
    plt.plot(data["t"], data["rotors_cmd"][:, 1], "r--")
    plt.ylim([-5.1, 40])

    plt.subplot(613, sharex=ax)
    plt.plot(data["t"], data["rotors"][:, 2], "k-")
    plt.plot(data["t"], data["rotors_cmd"][:, 2], "r--")
    plt.ylim([-5.1, 40])

    plt.subplot(614, sharex=ax)
    plt.plot(data["t"], data["rotors"][:, 3], "k-")
    plt.plot(data["t"], data["rotors_cmd"][:, 3], "r--")
    plt.ylim([-5.1, 40])

    # plt.subplot(615, sharex=ax)
    # plt.plot(data["t"], data["rotors"][:, 4], "k-")
    # plt.plot(data["t"], data["rotors_cmd"][:, 4], "r--")
    # plt.ylim([-5.1, 40])

    # plt.subplot(616, sharex=ax)
    # plt.plot(data["t"], data["rotors"][:, 5], "k-")
    # plt.plot(data["t"], data["rotors_cmd"][:, 5], "r--")
    # plt.ylim([-5.1, 40])

    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Rotor force")
    plt.tight_layout()

    plt.figure()

    ax = plt.subplot(411)
    plt.plot(data["t"], data["ref"][:, 0, 0], "r-", label="x (cmd)")
    plt.plot(data["t"], data["x"]["pos"][:, 0, 0], label="x")

    plt.plot(data["t"], data["ref"][:, 1, 0], "r--", label="y (cmd)")
    plt.plot(data["t"], data["x"]["pos"][:, 1, 0], label="y")

    plt.plot(data["t"], data["ref"][:, 2, 0], "r-.", label="z (cmd)")
    plt.plot(data["t"], data["x"]["pos"][:, 2, 0], label="z")

    plt.subplot(412, sharex=ax)
    plt.plot(data["t"], data["x"]["vel"].squeeze())
    plt.legend([r'$u$', r'$v$', r'$w$'])
    plt.subplot(413, sharex=ax)
    plt.plot(data["t"], data["x"]["quat"].squeeze())
    plt.legend([r'$q0$', r'$q1$', r'$q2$', r'$q3$'])
    # plt.plot(data["t"], np.transpose(quat2angle(np.transpose(data["x"]["quat"].squeeze()))))
    # plt.legend([r'$psi$', r'$theta$', r'$phi$'])
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

    fig3 = plt.figure()
    ax1 = fig3.add_subplot(4, 1, 1)
    ax2 = fig3.add_subplot(4, 1, 2, sharex=ax1)
    ax3 = fig3.add_subplot(4, 1, 3, sharex=ax1)
    ax4 = fig3.add_subplot(4, 1, 4, sharex=ax1)

    ax1.plot(data['t'], data['gamma'].squeeze()[:, 0])
    ax2.plot(data['t'], data['gamma'].squeeze()[:, 1])
    ax3.plot(data['t'], data['gamma'].squeeze()[:, 2])
    ax4.plot(data['t'], data['gamma'].squeeze()[:, 3])

    ax1.set_ylabel('gamma1')
    ax1.grid(True)
    ax2.set_ylabel('gamma2')
    ax2.grid(True)
    ax3.set_ylabel('gamma3')
    ax3.grid(True)
    ax4.set_ylabel('gamma4')
    ax4.grid(True)
    ax4.set_xlabel('Time [sec]')

    plt.tight_layout()

    # fig4 = plt.figure()
    # ax1 = fig4.add_subplot(4, 1, 1)
    # ax2 = fig4.add_subplot(4, 1, 2, sharex=ax1)
    # ax3 = fig4.add_subplot(4, 1, 3, sharex=ax1)
    # ax4 = fig4.add_subplot(4, 1, 4, sharex=ax1)

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

    # plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    exp1()
    exp1_plot()
