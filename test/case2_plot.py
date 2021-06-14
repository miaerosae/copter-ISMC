import matplotlib.pyplot as plt

import fym.logging


def case1_plot():
    data_A = fym.logging.load("case2_A.h5")
    data_I = fym.logging.load("case2_I.h5")
    data_L = fym.logging.load("case2_L.h5")

    plt.figure()

    ax = plt.subplot(311)
    plt.plot(data_A["t"], data_A["ref"][:, 0, 0], "r-", label="Desired x")
    plt.plot(data_A["t"], data_A["x"]["pos"][:, 0, 0], "--", label="AISMCA")
    plt.plot(data_I["t"], data_I["x"]["pos"][:, 0, 0], ":", label="ISMCA")
    plt.plot(data_L["t"], data_L["plant"]["pos"][:, 0, 0], "-.", label="LQRCA")
    plt.ylabel("position x [m]")
    plt.legend(loc='lower left')
    plt.axvspan(5, 5.042, alpha=0.2, color="b")
    plt.axvline(5.042, alpha=0.8, color="b", linewidth=0.5)
    plt.annotate("Rotor 0 : 0.1", xy=(5, -4.5), xytext=(5.5, -4.5),
                 arrowprops=dict(arrowstyle='->', lw=1.5))

    plt.subplot(312, sharex=ax)
    plt.plot(data_A["t"], data_A["ref"][:, 1, 0], "r-", label="Desired y")
    plt.plot(data_A["t"], data_A["x"]["pos"][:, 1, 0], "--", label="AISMCA")
    plt.plot(data_I["t"], data_I["x"]["pos"][:, 1, 0], ":", label="ISMCA")
    plt.plot(data_L["t"], data_L["plant"]["pos"][:, 1, 0], "-,", label="LQRCA")
    plt.ylabel("position y [m]")
    plt.legend(loc='lower left')
    plt.axvspan(5, 5.042, alpha=0.2, color="b")
    plt.axvline(5.042, alpha=0.8, color="b", linewidth=0.5)
    plt.annotate("Rotor 0 : 0.1", xy=(5, 50), xytext=(5.5, 50),
                 arrowprops=dict(arrowstyle='->', lw=1.5))

    plt.subplot(313, sharex=ax)
    plt.plot(data_A["t"], data_A["ref"][:, 2, 0], "r-", label="Desired z")
    plt.plot(data_A["t"], data_A["x"]["pos"][:, 2, 0], "--", label="AISMCA")
    plt.plot(data_I["t"], data_I["x"]["pos"][:, 2, 0], ":", label="ISMCA")
    plt.plot(data_L["t"], data_L["plant"]["pos"][:, 2, 0], "-,", label="LQRCA")
    plt.ylabel("position z [m]")
    plt.legend(loc='lower left')
    plt.axvspan(5, 5.042, alpha=0.2, color="b")
    plt.axvline(5.042, alpha=0.8, color="b", linewidth=0.5)
    plt.annotate("Rotor 0 : 0.1", xy=(5, 250), xytext=(5.5, 250),
                 arrowprops=dict(arrowstyle='->', lw=1.5))
    plt.xlabel("Time [sec]")

    plt.tight_layout()

    # for detail view
    plt.figure()

    ax = plt.subplot(311)
    plt.plot(data_A["t"], data_A["ref"][:, 0, 0], "r-", label="Desired x")
    plt.plot(data_A["t"], data_A["x"]["pos"][:, 0, 0], "--", label="AISMCA")
    plt.plot(data_I["t"], data_I["x"]["pos"][:, 0, 0], ":", label="ISMCA")
    plt.plot(data_L["t"], data_L["plant"]["pos"][:, 0, 0], "-.", label="LQRCA")
    plt.ylim([-1, 1.5])
    plt.ylabel("position x [m]")
    plt.legend(loc='lower left')
    plt.axvspan(5, 5.042, alpha=0.2, color="b")
    plt.axvline(5.042, alpha=0.8, color="b", linewidth=0.5)
    plt.annotate("Rotor 0 : 0.1", xy=(5, 0), xytext=(5.5, 0),
                 arrowprops=dict(arrowstyle='->', lw=1.5))

    plt.subplot(312, sharex=ax)
    plt.plot(data_A["t"], data_A["ref"][:, 1, 0], "r-", label="Desired y")
    plt.plot(data_A["t"], data_A["x"]["pos"][:, 1, 0], "--", label="AISMCA")
    plt.plot(data_I["t"], data_I["x"]["pos"][:, 1, 0], ":", label="ISMCA")
    plt.plot(data_L["t"], data_L["plant"]["pos"][:, 1, 0], "-,", label="LQRCA")
    plt.ylim([-2, 0.5])
    plt.ylabel("position y [m]")
    plt.legend(loc='lower left')
    plt.axvspan(5, 5.042, alpha=0.2, color="b")
    plt.axvline(5.042, alpha=0.8, color="b", linewidth=0.5)
    plt.annotate("Rotor 0 : 0.1", xy=(5, -0.5), xytext=(5.5, -0.5),
                 arrowprops=dict(arrowstyle='->', lw=1.5))

    plt.subplot(313, sharex=ax)
    plt.plot(data_A["t"], data_A["ref"][:, 2, 0], "r-", label="Desired z")
    plt.plot(data_A["t"], data_A["x"]["pos"][:, 2, 0], "--", label="AISMCA")
    plt.plot(data_I["t"], data_I["x"]["pos"][:, 2, 0], ":", label="ISMCA")
    plt.plot(data_L["t"], data_L["plant"]["pos"][:, 2, 0], "-,", label="LQRCA")
    plt.ylim([-3, 0.5])
    plt.ylabel("position z [m]")
    plt.legend(loc='lower left')
    plt.axvspan(5, 5.042, alpha=0.2, color="b")
    plt.axvline(5.042, alpha=0.8, color="b", linewidth=0.5)
    plt.annotate("Rotor 0 : 0.1", xy=(5, -1), xytext=(5.5, -1),
                 arrowprops=dict(arrowstyle='->', lw=1.5))
    plt.xlabel("Time [sec]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    case1_plot()
