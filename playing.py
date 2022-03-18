import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

actor_list=[]

try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town02')
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.find("vehicle.tesla.cybertruck")
    print(vehicle_bp)

    start_point = 1

    vehicle = world.spawn_actor(vehicle_bp, start_point)
    vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))
    actor_list.append(vehicle)
    time.sleep(20)


finally:
    for actor in actor_list:
        actor.destroy()
    print(("Clean up complete!"))