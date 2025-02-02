import numpy as np
from math import atan, sin, cos, pi
from matplotlib import pyplot as plt

class motion_primitive():
    """
    A class for generating smooth trajectories between two poses using cubic polynomial interpolation.

    This class generates motion primitives based on given boundary conditions for position and orientation.
    It computes the coefficients for cubic polynomials that describe the trajectories of x, y, and theta.
    
    Attributes:
        Ti (float): Initial time (default 0).
        Tf (float): Final time (default 1).
        Vr (float): Reference velocity (default 1).
        l (float): Car length (used for steering computation, default 2).
        theta_i (float): Initial heading (radians).
        theta_f (float): Final heading (radians).
        xi (float): Initial x-coordinate.
        xf (float): Final x-coordinate.
        yi (float): Initial y-coordinate.
        yf (float): Final y-coordinate.
        xi_dot (float): Initial derivative of x computed from Vr and theta_i.
        xf_dot (float): Final derivative of x computed from Vr and theta_f.
        yi_dot (float): Initial derivative of y computed from Vr and theta_i.
        yf_dot (float): Final derivative of y computed from Vr and theta_f.
    """

    def __init__(self, theta_i, theta_f, xi, xf, yi, yf):
        """
        Initialize the motion primitive with boundary conditions for position and orientation.

        Args:
            theta_i (float): Initial heading in radians.
            theta_f (float): Final heading in radians.
            xi (float): Initial x-coordinate.
            xf (float): Final x-coordinate.
            yi (float): Initial y-coordinate.
            yf (float): Final y-coordinate.
        """
        self.Ti = 0
        self.Tf = 1
        self.Vr = 1
        self.l = 2  # Car length (manually set)

        # Orientation values
        self.theta_i = theta_i
        self.theta_f = theta_f

        # X coordinate and its derivative
        self.xi = xi
        self.xi_dot = self.Vr * cos(theta_i)
        self.xf = xf
        self.xf_dot = self.Vr * cos(theta_f)

        # Y coordinate and its derivative
        self.yi = yi
        self.yi_dot = self.Vr * sin(theta_i)
        self.yf = yf
        self.yf_dot = self.Vr * sin(theta_f)

    def cubic_T_Matrix(self):
        """
        Construct the cubic time matrix and its inverse for trajectory generation.

        The time matrix is built using the boundary times Ti and Tf. Its inverse is computed to solve
        for the polynomial coefficients given the boundary conditions.
        
        Sets:
            self.T_matrix (np.matrix): The 4x4 time matrix.
            self.T_inv (np.matrix): The inverse of the time matrix.
        """
        self.T_matrix = np.matrix([
            [1, self.Ti, self.Ti ** 2, self.Ti ** 3],
            [0, 1, 2 * self.Ti, 3 * self.Ti ** 2],
            [1, self.Tf, self.Tf ** 2, self.Tf ** 3],
            [0, 1, 2 * self.Tf, 3 * self.Tf ** 2]
        ])
        self.T_inv = np.linalg.inv(self.T_matrix)

    def cubic_trajectory(self, X_matrix):
        """
        Compute the coefficients of the cubic polynomial for a given set of boundary conditions.

        Args:
            X_matrix (np.matrix): A 4x1 matrix containing the boundary conditions (position and derivative).

        Returns:
            np.matrix: The computed coefficients for the cubic polynomial.
        """
        Co_ef = np.dot(self.T_inv, X_matrix)
        return Co_ef

    def trajectory(self):
        """
        Compute the cubic trajectories for orientation (theta) and positions (x and y).

        This method calculates and sets the coefficients for the cubic polynomials that define the trajectory.
        
        Sets:
            self.theta_Co_ef (np.matrix): Coefficients for the theta (orientation) trajectory.
            self.x_Co_ef (np.matrix): Coefficients for the x-coordinate trajectory.
            self.y_Co_ef (np.matrix): Coefficients for the y-coordinate trajectory.
        """
        self.theta_matrix = np.matrix([[self.theta_i], [0], [self.theta_f], [0]])
        self.theta_Co_ef = self.cubic_trajectory(self.theta_matrix)

        self.x_matrix = np.matrix([[self.xi], [self.xi_dot], [self.xf], [self.xf_dot]])
        self.y_matrix = np.matrix([[self.yi], [self.yi_dot], [self.yf], [self.yf_dot]])

        self.x_Co_ef = self.cubic_trajectory(self.x_matrix)
        self.y_Co_ef = self.cubic_trajectory(self.y_matrix)

    def get_state(self, T):
        """
        Evaluate the trajectory state at a given time T.

        Args:
            T (float): Time at which to evaluate the trajectory (between Ti and Tf).

        Returns:
            tuple: A tuple containing (x, x_dot, y, y_dot, theta, theta_dot) at time T.
        """
        T_d1 = np.matrix([1, T, T**2, T**3])
        T_d2 = np.matrix([0, 1, 2 * T, 3 * T**2])

        x = np.dot(T_d1, self.x_Co_ef)
        x_dot = np.dot(T_d2, self.x_Co_ef)
        y = np.dot(T_d1, self.y_Co_ef)
        y_dot = np.dot(T_d2, self.y_Co_ef)
        theta = np.dot(T_d1, self.theta_Co_ef)
        theta_dot = np.dot(T_d2, self.theta_Co_ef)

        return x, x_dot, y, y_dot, theta, theta_dot

    def get_steering_angle(self, theta_dot, Vr):
        """
        Calculate the steering angle based on the angular velocity and vehicle length.

        Args:
            theta_dot (float): Angular velocity of the vehicle.
            Vr (float): Reference velocity.

        Returns:
            float: The computed steering angle.
        """
        return atan((theta_dot * self.l) / Vr)

    def get_path(self, step):
        """
        Generate a list of (x, y) points by sampling the trajectory at regular intervals.

        Args:
            step (float): The time increment used for sampling along the trajectory.

        Returns:
            tuple: Two lists containing the x and y coordinates of the trajectory.
        """
        i = 0
        pos_x = []
        pos_y = []
        while i < self.Tf - self.Ti:
            # Evaluate state at time i
            val = np.squeeze(np.asarray(self.get_state(i)))
            pos_x.append(val[0])
            pos_y.append(val[2])
            i += step
        return pos_x, pos_y

if __name__ == '__main__':
    # Example usage:
    # Create a motion primitive from initial heading 0 to final heading pi/2,
    # with starting position (1, 0) and ending position (2, 5)
    primitive = motion_primitive(0, pi/2, 1, 2, 0, 5)
    primitive.cubic_T_Matrix()
    primitive.trajectory()
    pos_x, pos_y = primitive.get_path(0.001)
    
    # Plot the trajectory
    plt.scatter(pos_x, pos_y)
    plt.title("Motion Primitive Trajectory")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.show()
