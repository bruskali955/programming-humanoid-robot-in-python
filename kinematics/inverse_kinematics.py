"""In this exercise you need to implement inverse kinematics for NAO's legs
* Tasks:
    1. solve inverse kinematics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    2. use the results of inverse kinematics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinematics implementation.
"""

from forward_kinematics import ForwardKinematicsAgent
from numpy import identity
from math import atan2
from numpy import sin, cos, pi, matrix, random, linalg, asarray
import numpy as np


def from_trans(m):
    thetaX = 0
    thetaY = 0
    thetaZ = 0

    if m[0, 0] == 1:
        thetaX = atan2(m[2, 1], m[1, 1])
    if m[1, 1] == 1:
        thetaY = atan2(m[0, 2], m[0, 0])
    if m[2, 2] == 1:
        thetaZ = atan2(m[1, 0], m[0, 0])
    return [m[3, 0], m[3, 1], m[3, 0], thetaX, thetaY, thetaZ]


class InverseKinematicsAgent(ForwardKinematicsAgent):
    def inverse_kinematics(self, effector_name, transform):
        """solve the inverse kinematics
        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        """

        # YOUR CODE HERE
        lambda_ = 1
        max_step = 0.1
        joint_angles = np.random.random(len(self.chains[effector_name]))

        target = np.matrix([from_trans(transform)]).T
        # Ts = [identity(len(self.chains[effector_name]))]
        for i in range(1000):
            Ts = [identity(len(self.chains[effector_name]))]
            for name in self.chains[effector_name]:
                Ts.append(self.transforms[name])

            Te = np.matrix([from_trans(Ts[-1])]).T

            error = target - Te
            error[error > max_step] = max_step
            error[error < -max_step] = -max_step
            T = np.matrix([from_trans(j) for j in Ts[0:-1]]).T
            J = Te - T
            dT = Te - T

            J[0, :] = dT[2, :]
            J[1, :] = dT[1, :]
            J[2, :] = dT[0, :]
            J[-1, :] = 1

            d_theta = lambda_ * np.linalg.pinv(J) * error
            joint_angles += np.asarray(d_theta.T)[0]
            if np.linalg.norm(d_theta) < 1e-4:
                break
        return joint_angles

    def set_transforms(self, effector_name, transform):
        """solve the inverse kinematics and control joints use the results
        """
        # YOUR CODE HERE

        angles = self.inverse_kinematics(effector_name, transform)
        names = []
        times = []
        keys = []
        for i, joint in enumerate(self.chains[effector_name]):
            names.append(joint)
            times.append([5.0, 7.0])
            keys.append([[angles[i], [2, 0.00000, 0.00000], [2, 0.00000, 0.00000]],
                         [angles[i], [1, 0.00000, 0.00000], [1, 0.00000, 0.00000]]])
        self.keyframes = (names, times, keys)


if __name__ == '__main__':
    agent = InverseKinematicsAgent()

    t = identity(4)
    t[-1, 1] = 0.05
    t[-1, 2] = 0.26
    agent.set_transforms('RArm', t)
    agent.run()
