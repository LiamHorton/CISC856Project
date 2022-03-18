import glob
import os
import sys

try:
    sys.path.append(glob.glob('PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

actor_list = []

try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10)
    #world = client.load_world('Town02')
    world = client.get_world()
    
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.cybertruck')
    print(vehicle_bp)
    
    spawn_start = 1

    transform = carla.Transform(carla.Location(x=220, y=200, z=20), carla.Rotation(yaw=180))
    vehicle = world.spawn_actor(vehicle_bp, transform)
    actor_list.append(vehicle)
    
finally:
    for actor in actor_list:
        actor.destroy()
    print("Cleaned up actors")
