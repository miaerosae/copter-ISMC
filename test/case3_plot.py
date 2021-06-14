import matplotlib.pyplot as plt

import fym.logging


def case3_plot():
    data_A = fym.logging.load("case3_A.h5")
    data_I = fym.logging.load("case3_I.h5")

    plt.figure()

    ax = plt.subplot(311)
    plt.plot(data_A["t"], data_A["ref"][:, 0, 0], "r-", label="Desired x")
    plt.plot(data_A["t"], data_A["x"]["pos"][:, 0, 0], "--", label="AISMCA")
    plt.plot(data_I["t"], data_I["x"]["pos"][:, 0, 0], ":", label="ISMCA")
    plt.ylabel("position x [m]")
    plt.legend(loc='lower left')
    plt.axvspan(5, 5.042, alpha=0.2, color="b")
    plt.axvline(5.042, alpha=0.8, color="b", linewidth=0.5)
    plt.annotate("Rotor 0 fault", xy=(5, -50), xytext=(5.5, -50),
                 arrowprops=dict(arrowstyle='->', lw=1.5))

    plt.subplot(312, sharex=ax)
    plt.plot(data_A["t"], data_A["ref"][:, 1, 0], "r-", label="Desired y")
    plt.plot(data_A["t"], data_A["x"]["pos"][:, 1, 0], "--", label="AISMCA")
    plt.plot(data_I["t"], data_I["x"]["pos"][:, 1, 0], ":", label="ISMCA")
    plt.ylabel("position y [m]")
    plt.legend(loc='lower left')
    plt.axvspan(5, 5.042, alpha=0.2, color="b")
    plt.axvline(5.042, alpha=0.8, color="b", linewidth=0.5)
    plt.annotate("Rotor 0 : fault", xy=(5, 200), xytext=(5.5, 200),
                 arrowprops=dict(arrowstyle='->', lw=1.5))

    plt.subplot(313, sharex=ax)
    plt.plot(data_A["t"], data_A["ref"][:, 2, 0], "r-", label="Desired z")
    plt.plot(data_A["t"], data_A["x"]["pos"][:, 2, 0], "--", label="AISMCA")
    plt.plot(data_I["t"], data_I["x"]["pos"][:, 2, 0], ":", label="ISMCA")
    plt.ylabel("position z [m]")
    plt.legend(loc='lower left')
    plt.axvspan(5, 5.042, alpha=0.2, color="b")
    plt.axvline(5.042, alpha=0.8, color="b", linewidth=0.5)
    plt.annotate("Rotor 0 fault", xy=(5, 400), xytext=(5.5, 400),
                 arrowprops=dict(arrowstyle='->', lw=1.5))
    plt.xlabel("Time [sec]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    case3_plot()
