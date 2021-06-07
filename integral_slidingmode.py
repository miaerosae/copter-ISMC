import numpy as np
from numpy import cos

from fym.core import BaseEnv, BaseSystem


def sat(s, eps):
    if s > eps:
        return 1
    elif s < -eps:
        return -1
    else:
        return s/eps


class IntegralSMC(BaseEnv):
    '''
    reference
    Ban Wang, Youmin Zhang, "An Adaptive Fault-Tolerant Sliding Mode Control
    Allocation Scheme for Multirotor Helicopter Subjected to
    Actuator Faults", IEEE Transactions on industrial electronics, Vol. 65,
    No. 5, May 2018
    '''

    def __init__(self, J, m, g, d, ic, ref0):
        super().__init__()
        self.ic = ic
        self.ref0 = ref0
        self.J, self.m, self.g, self.d = J, m, g, d
        self.P = BaseSystem(np.vstack((self.ic[2] - self.ref0[2],
                                       self.ic[6] - self.ref0[6],
                                       self.ic[7] - self.ref0[7],
                                       self.ic[8] - self.ref0[8])))

    def deriv(self, obs, ref):
        # observation
        obs = np.vstack((obs))
        z = obs[2]
        phi, theta, psi = obs[6:9]
        # reference
        ref = np.vstack((ref))
        z_r = ref[2]
        phi_r, theta_r, psi_r = ref[6:9]
        dP = np.vstack((z - z_r,
                        phi - phi_r,
                        theta - theta_r,
                        psi - psi_r))

        return dP

    def set_dot(self, obs, ref):
        dot = self.deriv(obs, ref)
        self.P.dot = dot

    def get_FM(self, obs, ref, p, K, Kc, PHI, t):
        p = np.vstack((p))
        p1, p2, p3, p4 = p
        K1, K2, K3, K4 = K
        k11, k12 = K1
        k21, k22 = K2
        k31, k32 = K3
        k41, k42 = K4
        kc1, kc2, kc3, kc4 = Kc
        PHI1, PHI2, PHI3, PHI4 = PHI
        # model
        J = self.J
        Ixx = J[0, 0]
        Iyy = J[1, 1]
        Izz = J[2, 2]
        m, g, d = self.m, self.g, self.d
        # observation
        obs = np.vstack((obs))
        x, y, z, xd, yd, zd = obs[0:6]
        phi, theta, psi, phid, thetad, psid = obs[6:]
        # reference
        x_r, y_r, z_r, xd_r, yd_r, zd_r = ref[0:6]
        phi_r, theta_r, psi_r, phid_r, thetad_r, psid_r = ref[6:]
        zdd_r = 0
        phidd_r = 0
        thetadd_r = 0
        psidd_r = 0
        # initial condition
        z0, z0d = self.ic[2], self.ic[5]
        phi0, theta0, psi0, phi0d, theta0d, psi0d = self.ic[6:]
        z0_r, z0d_r = self.ref0[2], self.ref0[5]
        phi0_r, theta0_r, psi0_r, phi0d_r, theta0d_r, psi0d_r = self.ref0[6:]
        # PD control for position tracking (get phi_ref, theta_ref)
        e_x = x - x_r
        e_xd = xd - xd_r
        e_y = y - y_r
        e_yd = yd - yd_r
        theta_r = (0.19*e_x + 0.2*e_xd)
        phi_r = -(0.19*e_y + 0.2*e_yd)
        # error definition
        e_z = z - z_r
        e_zd = zd - zd_r
        e_phi = phi - phi_r
        e_phid = phid - phid_r
        e_theta = theta - theta_r
        e_thetad = thetad - thetad_r
        e_psi = psi - psi_r
        e_psid = psid - psid_r
        # h**(-1) function definition
        h1 = -m/cos(phi)/cos(theta)
        h2 = Ixx/d
        h3 = Iyy/d
        h4 = Izz/d
        # sliding surface
        s1 = e_zd + k12*e_z + k11*p1 - k12*(z0-z0_r) - (z0d-z0d_r)
        s2 = e_phid + k22*e_phi + k21*p2 - k22*(phi0-phi0_r) - (phi0d-phi0d_r)
        s3 = e_thetad + k32*e_theta + k31*p3 - k32*(theta0-theta0_r) \
            - (theta0d-theta0d_r)
        s4 = e_psid + k42*e_psi + k41*p4 - k42*(psi0-psi0_r) - (psi0d-psi0d_r)
        # get FM
        F = h1*(zdd_r - k12*e_zd - k11*e_z - g) - h1*kc1*sat(s1, PHI1)
        M1 = h2*(phidd_r - k22*e_phid - k21*e_phi - (Iyy-Izz)/Ixx*thetad*psid) \
            - h2*kc2*sat(s2, PHI2)
        M2 = h3*(thetadd_r - k32*e_thetad - k31*e_theta
            - (Izz-Ixx)/Iyy*phid*psid) - h3*kc3*sat(s3, PHI3)
        M3 = h4*(psidd_r - k42*e_psid - k41*e_psi - (Ixx-Iyy)/Izz*phid*thetad) \
            - h4*kc4*sat(s4, PHI4)

        action = np.vstack((F, M1, M2, M3))
        sliding = np.vstack((s1, s2, s3, s4))
        return action, sliding


if __name__ == "__main__":
    pass
