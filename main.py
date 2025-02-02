import glob
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import carla
import numpy as np
import pygame
import math

# Append the CARLA egg file path (adjust if needed)
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    import carla
except IndexError:
    pass

# Import your own modules
from motion_primitve import motion_primitive
from controller import VehiclePIDController
from RRT import RRT

# =============================================================================
# PYGAME RENDERING (Camera view)
# =============================================================================
class RenderObject(object):
    def __init__(self, width, height):
        # Create a random image to initialize the surface
        init_image = np.random.randint(0, 255, (height, width, 3), dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0, 1))


def pygame_callback(data, obj):
    # Convert raw data to an image and update the surface
    img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
    img = img[:, :, :3]
    img = img[:, :, ::-1]
    obj.surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))


# =============================================================================
# WORLD CLASS: Connect and configure the CARLA world
# =============================================================================
class world():
    def __init__(self, Town):
        # Connect to CARLA and load the desired town
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(120.0)
        self.carla_world = self.client.load_world(Town)
        self.map = self.carla_world.get_map()
        print("WORLD READY")

        # Set simulator to synchronous mode with a fixed time step
        settings = self.carla_world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.carla_world.apply_settings(settings)

        # Set spectator so you can view the simulation
        self.spectator = self.carla_world.get_spectator()

        self.vehicles = []
        self.ego_vehicle = None


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    # Create and set up the world
    CARLA_world = world('Town03')
    spawn_points = CARLA_world.map.get_spawn_points()
    blueprint_library = CARLA_world.carla_world.get_blueprint_library()

    # -----------------------------
    # Spawn Obstacle Vehicles
    # -----------------------------
    spawn_point1 = carla.Transform(carla.Location(x=2.3, y=140, z=0.3),
                                   carla.Rotation(yaw=-90))
    bp1 = blueprint_library.filter("model3")[0]
    vehicle1 = CARLA_world.carla_world.spawn_actor(bp1, spawn_point1)
    CARLA_world.vehicles.append(vehicle1)

    spawn_point2 = carla.Transform(carla.Location(x=6.3, y=120, z=0.3),
                                   carla.Rotation(yaw=-90))
    bp2 = blueprint_library.filter("model3")[0]
    vehicle2 = CARLA_world.carla_world.spawn_actor(bp2, spawn_point2)
    CARLA_world.vehicles.append(vehicle2)

    spawn_point4 = carla.Transform(carla.Location(x=1.3, y=100, z=0.3),
                                   carla.Rotation(yaw=-90))
    bp4 = blueprint_library.filter("model3")[0]
    vehicle3 = CARLA_world.carla_world.spawn_actor(bp4, spawn_point4)
    CARLA_world.vehicles.append(vehicle3)
    
    spawn_point5 = carla.Transform(carla.Location(x=1.3, y=110, z=0.3),
                                   carla.Rotation(yaw=-90))
    bp5 = blueprint_library.filter("model3")[0]
    vehicle5 = CARLA_world.carla_world.spawn_actor(bp5, spawn_point5)
    CARLA_world.vehicles.append(vehicle5)

    # -----------------------------
    # Spawn the Ego Vehicle
    # -----------------------------
    spawn_point3 = carla.Transform(carla.Location(x=2.3, y=160, z=0.3),
                                   carla.Rotation(yaw=-90))
    bp3 = blueprint_library.filter("model3")[0]
    CARLA_world.ego_vehicle = CARLA_world.carla_world.spawn_actor(bp3, spawn_point3)

    # -----------------------------
    # Attach a Camera Sensor to the Ego Vehicle
    # -----------------------------
    camera_init_trans = carla.Transform(carla.Location(x=-5, z=3), 
                                        carla.Rotation(pitch=-20))
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera = CARLA_world.carla_world.spawn_actor(camera_bp, camera_init_trans,
                                                 attach_to=CARLA_world.ego_vehicle)
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()

    renderObject = RenderObject(image_w, image_h)
    camera.listen(lambda image: pygame_callback(image, renderObject))

    # Initialize PyGame display for camera view
    pygame.init()
    gameDisplay = pygame.display.set_mode((image_w, image_h), pygame.HWSURFACE | pygame.DOUBLEBUF)
    gameDisplay.fill((0, 0, 0))
    gameDisplay.blit(renderObject.surface, (0, 0))
    pygame.display.flip()

    # -----------------------------
    # Define Start and Goal for Planning
    # -----------------------------
    goal = CARLA_world.map.get_waypoint(carla.Location(x=2.3, y=80), project_to_road=True)
    start = CARLA_world.map.get_waypoint(carla.Location(x=2.3, y=160), project_to_road=True)
    print("Goal:", goal.transform.location.x, goal.transform.location.y)
    CARLA_world.carla_world.tick()

    # Get transforms of obstacles (vehicles) to be used in planning
    trans = CARLA_world.vehicles[0].get_transform()
    trans2 = CARLA_world.vehicles[1].get_transform()
    trans3 = CARLA_world.vehicles[2].get_transform()
    obstacles = [trans, trans2, trans3]
    print("Obstacle 1:", trans.location, trans.rotation.yaw)

    # -----------------------------
    # Run RRT* Planner to get a rough path (list of waypoints)
    # -----------------------------
    RRT_planner = RRT(CARLA_world, goal, obstacles)
    RRT_planner.RRT_star(n_pts=1000)
    path = RRT_planner.path

    # -----------------------------
    # Generate a Smooth Trajectory Using Motion Primitives
    # -----------------------------
    path_x = []
    path_y = []
    trans = CARLA_world.ego_vehicle.get_transform()
    thetai = trans.rotation.yaw * math.pi / 180  # convert to radians
    final_theta = thetai
    print("Initial heading (rad):", thetai)
    for i in range(len(path) - 1):
        if i == len(path) - 2:
            thetaf = final_theta
        else:
            x1, y1 = path[i + 1].x, path[i + 1].y
            x2, y2 = path[i + 2].x, path[i + 2].y
            thetaf = math.atan2((y2 - y1), (x2 - x1))
        primitive = motion_primitive(thetai, thetaf, path[i].x, path[i + 1].x, path[i].y, path[i + 1].y)
        primitive.cubic_T_Matrix()
        primitive.trajectory()
        pos_x, pos_y = primitive.get_path(0.05)
        path_x += pos_x
        path_y += pos_y
        thetai = thetaf

    print("Smooth trajectory X:", path_x)
    print("Smooth trajectory Y:", path_y)

    # -----------------------------
    # Set Up Real-Time Matplotlib Visualization
    # -----------------------------
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Real-Time Trajectory")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    # Plot the planned trajectory once (green)
    line_planned, = ax.plot(path_x, path_y, 'g-', lw=2, label="Planned Trajectory")
    # Prepare empty plots for the actual trajectory (red) and current position (blue)
    line_actual, = ax.plot([], [], 'r-', lw=2, label="Actual Trajectory")
    point_car, = ax.plot([], [], 'bo', markersize=8, label="Ego Vehicle")
    ax.legend()
    plt.show()

    # -----------------------------
    # Control Loop: Follow the Trajectory
    # -----------------------------
    controller = VehiclePIDController(CARLA_world.ego_vehicle, [15, 5, 0], [5, 1, 0])
    actual_x = []
    actual_y = []

    # Loop over trajectory segments
    for i in range(len(path_x) - 1):
        # Get current target segment endpoints
        trans = CARLA_world.ego_vehicle.get_transform()
        w_x, w_y = path_x[i], path_y[i]
        w_x2, w_y2 = path_x[i+1], path_y[i+1]
        # Compute heading difference (will be used later)
        phi = math.atan2((w_y2 - w_y), (w_x2 - w_x))
        physics = CARLA_world.ego_vehicle.get_physics_control()
        wheels = physics.wheels
        # Approximate front wheel position (this example averages two front wheels)
        wheel_F_x = (wheels[0].position.x + wheels[1].position.x) / 200
        wheel_F_y = (wheels[0].position.y + wheels[1].position.y) / 200
        print("------ New Segment ------")
        # Continue applying control until close enough to the target segment endpoint
        while math.sqrt((wheel_F_x - w_x2)**2 + (wheel_F_y - w_y2)**2) >= 0.3:
            # Get control from longitudinal PID controller
            control = controller.run_step(10)
            print("Target:", w_x2, w_y2)
            print("Wheel position:", wheel_F_x, wheel_F_y)
            # Compute error for lateral correction
            p1 = np.array([w_x, w_y])
            p2 = np.array([w_x2, w_y2])
            p3 = np.array([wheel_F_x, wheel_F_y])
            trans = CARLA_world.ego_vehicle.get_transform()
            yaw = trans.rotation.yaw
            # Adjust heading error based on current yaw
            phi = math.atan2((w_y2 - w_y), (w_x2 - w_x)) - yaw * (math.pi/180)
            # Cross-track error (lateral deviation)
            d = np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1)
            kp = 6
            ks = 0.2
            Vel = CARLA_world.ego_vehicle.get_velocity()
            v = math.sqrt(Vel.x**2 + Vel.y**2)
            control.steer = (-math.atan2(kp * d, ks + v) + phi)
            CARLA_world.ego_vehicle.apply_control(control)
            CARLA_world.carla_world.tick()
            physics = CARLA_world.ego_vehicle.get_physics_control()
            wheels = physics.wheels
            wheel_F_x = (wheels[0].position.x + wheels[1].position.x) / 200
            wheel_F_y = (wheels[0].position.y + wheels[1].position.y) / 200

            # Update the PyGame display (camera view)
            gameDisplay.fill((0, 0, 0))
            gameDisplay.blit(renderObject.surface, (0, 0))
            pygame.display.flip()

            # Record the current wheel (approximate vehicle) position
            actual_x.append(wheel_F_x)
            actual_y.append(wheel_F_y)

            # -----------------------------
            # Update the Real-Time Matplotlib Plot
            # -----------------------------
            line_actual.set_data(actual_x, actual_y)
            point_car.set_data(wheel_F_x, wheel_F_y)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

    # -----------------------------
    # Final Plot (blocking) once simulation is complete
    # -----------------------------
    plt.ioff()  # Turn interactive plotting off
    plt.figure()
    plt.plot([start.transform.location.x], [start.transform.location.y],
             marker='o', markersize=9, label="Start")
    plt.text(start.transform.location.x, start.transform.location.y, "Start")
    plt.plot([goal.transform.location.x], [goal.transform.location.y],
             marker='o', markersize=9, label="Goal")
    plt.text(goal.transform.location.x, goal.transform.location.y, "Goal")
    plt.plot(actual_x, actual_y, 'r-', label="Actual Trajectory")
    plt.legend()
    plt.title("Final Trajectory")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.show()

    # Cleanup
    camera.stop()
    pygame.quit()
