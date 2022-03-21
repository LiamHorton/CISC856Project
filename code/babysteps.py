import glob
import os
from pickle import TRUE
from queue import Queue
import sys

from zmq import QUEUE

try:
    sys.path.append(glob.glob('../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


import carla
from queue import Queue
from queue import Empty
from PIL import Image
import numpy as np

def sensor_callback(data, queue):
    """
    This simple callback just stores the data on a thread safe Python Queue
    to be retrieved from the "main thread".
    """
    queue.put(data)

def run_simulation(client):

    try:
        sim_time = 20
        time_step = 0.1
        
        # Get the world and its information
        world = client.get_world()
        bp_lib = world.get_blueprint_library()
        original_settings = world.get_settings()

        # Configure the world
        settings = world.get_settings()
        settings.synchronous_mode = True #make the server wait for the client
        settings.fixed_delta_seconds = time_step
        world.apply_settings(settings)
      
        vehicle = None
        camera = None
        collision = None
        
        # Get the required blueprints
        vehicle_bp = bp_lib.filter('cybertruck')[0]
        camera_bp = bp_lib.filter('sensor.camera.rgb')[0]

        # # Configure the blueprints
        camera_bp.set_attribute("image_size_x", '200')
        camera_bp.set_attribute("image_size_y", '150')
        collision_bp = world.get_blueprint_library().find('sensor.other.collision')
        # Consider adding noise and and blurring with enable_postprocess_effects
        
        

        # Spawn our actors
        vehicle = world.spawn_actor(blueprint=vehicle_bp, transform=world.get_map().get_spawn_points()[0])
        camera = world.spawn_actor(blueprint=camera_bp, transform=carla.Transform(carla.Location(x=1.6, z=1.6)), attach_to=vehicle)
        collision = world.spawn_actor(blueprint=collision_bp, transform=carla.Transform(), attach_to=vehicle)

        image_queue = Queue()
        camera.listen(lambda data: sensor_callback(data, image_queue))
        collision_queue = Queue()
        collision.listen(lambda data: sensor_callback(data, collision_queue))

        turn = False
        right_turn = False
        for step in range(int(sim_time/time_step)):
            if (step+1)%5==0:
                turn = not(turn)
                if turn:
                    right_turn = not(right_turn)
            
            if turn and right_turn:
                vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.5))
            elif turn:
                vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=-0.5))
            else:
                vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))
            

            world.tick()
            world_frame = world.get_snapshot().frame

        
            try:
                image_data = image_queue.get(True, 1.0)
            except Empty:
                print("[Warning] Some sensor data has been missed")
                continue

            #collision_data = collision_queue.get(True, 1.0)
        
            #output information to the screen
            sys.stdout.write("\r(%d/%d) Simulation: %d Camera: %d" %
                (step+1, sim_time/time_step, world_frame, image_data.frame) + ' ')
            sys.stdout.flush()

            im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
            im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
            im_array = im_array[:, :, :3][:, :, ::-1]

            # #Save the image using Pillow module.
            # image = Image.fromarray(im_array)
            # image.save("../output_data/%08d.png" % image_data.frame)

        #print(collision_data)
    
    finally:
        # Apply the original settings when exiting.
        world.apply_settings(original_settings)

        # Destroy the actors in the scene.
        if camera:
            camera.destroy()
        if vehicle:
            vehicle.destroy()
        if collision:
            collision.destroy()



def main():
    
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        run_simulation(client)
    
    except:
        print("Something went wrong.")

if __name__ == '__main__':

    main()
