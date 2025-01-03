import bpy
import numpy as np
from mathutils import Vector
from numpy.random import uniform as U, normal as N, randint as RI

def get_corner_directions(resolution_x, resolution_y, f_in_mm, sensor_width_in_mm, sensor_height_in_mm):
    fx = f_in_mm / sensor_width_in_mm * resolution_x
    fy = f_in_mm / sensor_height_in_mm * resolution_y
    cx = resolution_x / 2.0
    cy = resolution_y / 2.0

    corners = []
    for i in [0, 0.5, 1]:
        for j in [0, 0.5, 1]:
            corners.append((resolution_x * i, resolution_y * j))
    directions = []
    for corner in corners:
        x, y = corner
        x = (x - cx) / fx
        y = (cy - y) / fy 
        directions.append(Vector((x, y, -1)))  
    
    return directions

def get_moving_threshold(config):
    if "moving_threshold" in config:
        moving_threshold = config["moving_threshold"]
    else:
        moving_threshold = [u - v for u, v in zip(config["camera_range"]["loc_upper_bound"], config["camera_range"]["loc_lower_bound"])]
        moving_threshold = sum(moving_threshold) / len(moving_threshold)
    return moving_threshold

def get_camera_sample(resolution_x, resolution_y, sensor_width, sensor_height, scene, args, config):
    def check_valid(camera_loc, camera_dir, lens):
        rotation = camera_dir.rotation_difference(Vector((0, 0, 1)))
        corner_directions = get_corner_directions(resolution_x, resolution_y, lens, sensor_width, sensor_height)
        world_directions = [rotation @ dir for dir in corner_directions]

        distances = []
        for world_dir in world_directions:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            result, location, normal, index, object, matrix = scene.ray_cast(depsgraph, camera_loc, world_dir)
            camera_lower = config['camera_range']['distance_lower_bound']
            camera_upper = config['camera_range']['distance_upper_bound']
            if result:
                distance = (Vector(location) - camera_loc).length
                distances.append(distance)
                if distance < camera_lower or (distance > camera_upper and camera_upper > 0):
                    return False, distances
            elif camera_upper > 0:
                return False, None

        return True, distances

    loc_lower_bound = config['camera_range']['loc_lower_bound']
    loc_upper_bound = config['camera_range']['loc_upper_bound']
    dir_lower_bound = config['camera_range']['dir_lower_bound']
    dir_upper_bound = config['camera_range']['dir_upper_bound']
    focus_lower_bound = config['camera_range']['focus_lower_bound']
    focus_upper_bound = config['camera_range']['focus_upper_bound']
    moving_threshold = get_moving_threshold(config)

    camera_coefs = []

    if args.sample_view:
        camera_loc = Vector(config['camera']['loc'])
        camera_dir = Vector(config['camera']['dir'])
        camera_dir.normalize()
        lens = config['camera']['len']
        camera_coefs.append((camera_loc, camera_dir, lens))
        # check_valid(camera_loc, camera_dir, lens)

    target = args.num_images
    trys = 10000
    while trys > 0:
        if len(camera_coefs) >= target:
            break
        camera_loc = Vector([np.random.uniform(loc_lower_bound[i], loc_upper_bound[i]) for i in range(3)])
        camera_dir = Vector([np.random.uniform(dir_lower_bound[i], dir_upper_bound[i]) for i in range(3)])
        perpendicular_vector = camera_dir.cross(Vector((0, 0, 1)))
        perpendicular_vector.normalize()
        camera2_offset = perpendicular_vector * 0.3
        camera2_loc = camera_loc + camera2_offset

        camera_dir.normalize()  
        lens = np.random.uniform(focus_lower_bound, focus_upper_bound) 
        result1, distances1 = check_valid(camera_loc, camera_dir, lens)
        result2, distances2 = check_valid(camera2_loc, camera_dir, lens)
        if result1 and result2:
            camera_coefs.append((camera_loc, camera_dir, lens))
        trys -= 1

    print(f"Found {len(camera_coefs)} valid camera samples")

    next_camera_coefs = []
    for camera_coef in camera_coefs:
        camera_loc, camera_dir, lens = camera_coef
        new_camera_loc = camera_loc + Vector([N(0, 0.03) for i in range(3)]) * moving_threshold
        new_camera_dir = camera_dir + Vector([N(0, 0.03) for i in range(3)])
        new_camera_dir.normalize()
        next_camera_coefs.append((new_camera_loc, new_camera_dir, lens))
    return camera_coefs, next_camera_coefs 

def setup_camera_frame(scene, camera, coef, frame):
    camera_location, camera_direction, lens = coef

    bpy.context.scene.frame_set(frame)
    camera.data.lens = lens
    camera.location = Vector(camera_location)
    camera.rotation_mode = 'XYZ'
    rot_quat = camera_direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    camera.keyframe_insert(data_path="location", frame=frame)
    camera.keyframe_insert(data_path="rotation_euler", frame=frame)

def setup_stereo_camera(scene, coef, next_coef, config):
    moving_threshold = get_moving_threshold(config)
    camera = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.stereo.convergence_distance = 1000
    camera.data.stereo.interocular_distance = moving_threshold / 20
    camera.data.stereo.convergence_mode = 'PARALLEL'

    setup_camera_frame(scene, camera, coef, 0)
    setup_camera_frame(scene, camera, next_coef, 1)
    return camera

def remove_all_camera(scene):
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)