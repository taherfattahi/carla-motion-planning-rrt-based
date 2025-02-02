from collections import deque
import math
import numpy as np
import sys
import glob
import os
import carla

# Append the CARLA egg file to the Python path
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    import carla
except IndexError:
    pass


def get_speed(vehicle):
    """
    Compute the speed of a vehicle in Km/h.

    Args:
        vehicle: A CARLA vehicle actor whose speed is to be calculated.

    Returns:
        float: The speed of the vehicle in kilometers per hour.
    """
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


class VehiclePIDController():
    """
    Combines longitudinal and lateral PID controllers to generate low-level control commands
    (throttle, brake, and potentially steer) for a vehicle.
    """

    def __init__(self, vehicle, args_lateral, args_longitudinal, offset=0, max_throttle=0.75, max_brake=0.3, max_steering=0.8):
        """
        Initialize the VehiclePIDController.

        Args:
            vehicle: The CARLA vehicle actor to control.
            args_lateral: A list of PID gains [K_P, K_D, K_I] for lateral control.
            args_longitudinal: A list of PID gains [K_P, K_D, K_I] for longitudinal control.
            offset (float): Lateral offset from the centerline (default 0).
            max_throttle (float): Maximum throttle command (default 0.75).
            max_brake (float): Maximum brake command (default 0.3).
            max_steering (float): Maximum steering command (default 0.8).
        """
        self.max_brake = max_brake
        self.max_throt = max_throttle
        self.max_steer = max_steering

        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self.past_steering = self._vehicle.get_control().steer
        self._lon_controller = PIDLongitudinalController(self._vehicle,
                                                         args_longitudinal[0],
                                                         args_longitudinal[1],
                                                         args_longitudinal[2])
        self._lat_controller = PIDLateralController(self._vehicle,
                                                    offset,
                                                    args_lateral[0],
                                                    args_lateral[1],
                                                    args_lateral[2])

    def run_step(self, target_speed):
        """
        Execute one control step to approach a target speed.

        This method uses the longitudinal PID controller to calculate the necessary acceleration
        (or deceleration) and then returns a VehicleControl command with appropriate throttle and brake values.
        (Note: Lateral (steering) control is available via the lateral controller, but it is currently commented out.)

        Args:
            target_speed (float): The desired vehicle speed in Km/h.

        Returns:
            carla.VehicleControl: The control command containing throttle and brake values.
        """
        acceleration = self._lon_controller.run_step(target_speed)
        control = carla.VehicleControl()
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)
        # Lateral control (steering) can be implemented by using:
        # current_steering = self._lat_controller.run_step(waypoint)
        # and then regulating steering changes before assigning it to control.steer.
        return control


class PIDLongitudinalController():
    """
    Implements longitudinal vehicle control using a PID controller.
    """

    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Initialize the PIDLongitudinalController.

        Args:
            vehicle: The CARLA vehicle actor.
            K_P (float): Proportional gain.
            K_D (float): Differential gain.
            K_I (float): Integral gain.
            dt (float): Time step (in seconds) for integration and differentiation.
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_speed, debug=False):
        """
        Compute the PID control action to drive the vehicle toward the target speed.

        Args:
            target_speed (float): The desired speed in Km/h.
            debug (bool): If True, prints debug information.

        Returns:
            float: The computed control value (positive for throttle, negative for brake).
        """
        current_speed = get_speed(self._vehicle)
        if debug:
            print('Current speed = {}'.format(current_speed))
        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Perform the internal PID calculation for longitudinal control.

        Args:
            target_speed (float): The desired speed in Km/h.
            current_speed (float): The current speed in Km/h.

        Returns:
            float: The PID controller output (clipped between -1 and 1).
        """
        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)


class PIDLateralController():
    """
    Implements lateral vehicle control using a PID controller.
    """

    def __init__(self, vehicle, offset=0, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Initialize the PIDLateralController.

        Args:
            vehicle: The CARLA vehicle actor.
            offset (float): Lateral offset from the center line. A nonzero value displaces the target waypoint.
            K_P (float): Proportional gain.
            K_D (float): Differential gain.
            K_I (float): Integral gain.
            dt (float): Time step (in seconds) for integration and differentiation.
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._offset = offset
        self._e_buffer = deque(maxlen=10)

    def run_step(self, waypoint):
        """
        Compute the lateral control action to steer the vehicle toward the target waypoint.

        Args:
            waypoint: The target waypoint (with transform information).

        Returns:
            float: Steering control value in the range [-1, 1] where -1 indicates maximum left steering and +1 indicates maximum right steering.
        """
        return self._pid_control(waypoint, self._vehicle.get_transform())

    def _pid_control(self, waypoint, vehicle_transform):
        """
        Perform the internal PID calculation for lateral control.

        This function computes the angular error between the vehicle's forward vector and the vector pointing from the vehicle
        to the target waypoint. It then applies a PID correction to compute the steering command.

        Args:
            waypoint: The target waypoint.
            vehicle_transform: The current transform of the vehicle.

        Returns:
            float: The computed steering command (clipped between -1 and 1).
        """
        # Get the vehicle's current position and forward vector.
        ego_loc = vehicle_transform.location
        v_vec = vehicle_transform.get_forward_vector()
        v_vec = np.array([v_vec.x, v_vec.y, 0.0])

        # Compute the target waypoint location, applying lateral offset if necessary.
        if self._offset != 0:
            w_tran = waypoint.transform
            r_vec = w_tran.get_right_vector()
            w_loc = w_tran.location + carla.Location(x=self._offset * r_vec.x,
                                                     y=self._offset * r_vec.y)
        else:
            w_loc = waypoint.transform.location

        # Create the vector from the vehicle to the waypoint.
        w_vec = np.array([w_loc.x - ego_loc.x,
                          w_loc.y - ego_loc.y,
                          0.0])

        # Compute the angular error between the vehicle's heading and the direction to the waypoint.
        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))
        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        # Update the error buffer and compute derivative and integral terms.
        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * _dot) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)
