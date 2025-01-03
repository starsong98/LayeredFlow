import bpy
import bmesh
import math
import random
import os
import string
import numpy as np
from mathutils import Vector
from .material import assign_glass_material, add_randomness_to_glass, assign_metalic_material, assign_rough_glass_material, is_obj_has_emission_material
from numpy.random import uniform as U, normal as N, randint as RI
from scipy.spatial.transform import Rotation as R
import sys

def perturb_rotation_matrix(matrix, noise_std_dev=0.3):
    matrix.x += N(0, noise_std_dev)
    matrix.y += N(0, noise_std_dev)
    matrix.z += N(0, noise_std_dev)
    return matrix

def get_object(counter, file_name='window_assets'):
    filepath = f"external_blends/assets/{file_name}.blend"

    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        collection_list = data_from.collections
        collection_name = random.choice(collection_list)
        data_to.collections.append(collection_name)
    
    random_string = ''.join(random.choices(string.ascii_lowercase, k=4))
    collection = bpy.data.collections.get(collection_name)
    collection.name = f"{file_name}_{random_string}_{counter}"
    bpy.context.scene.collection.children.link(collection)

    for obj in collection.objects:
        if obj.parent is None:
            return obj
    return None

def set_object_location_and_rotation(obj, location, rotation_euler, moving_threshold):
    bpy.context.scene.frame_set(0)
    obj.location = location
    obj.rotation_euler = rotation_euler
    obj.keyframe_insert(data_path="location", frame=0)

    bpy.context.scene.frame_set(1)
    obj.location = location + Vector((N(0, 0.02), N(0, 0.02), N(0, 0.02))) * moving_threshold
    obj.rotation_euler = perturb_rotation_matrix(rotation_euler, 0.05)
    obj.keyframe_insert(data_path="location", frame=1)

def get_cylinder_plane(size):
    bpy.ops.mesh.primitive_cylinder_add(radius=size / 2, depth=size, location=(0, 0, 0), vertices=1024)
    cylinder = bpy.context.active_object

    # Setup BMesh and ensure the mesh is updated
    bm = bmesh.new()
    bm.from_mesh(cylinder.data)

    # Delete top and bottom faces
    for face in bm.faces:
        if face.normal.z == 1.0 or face.normal.z == -1.0:
            bm.faces.remove(face)

    offset = U(0, math.pi / 4)
    start = offset  
    end = math.pi - offset

    for vert in list(bm.verts):
        angle = math.atan2(vert.co.y, vert.co.x)
        if not (start <= angle and angle <= end):
            bm.verts.remove(vert)

    # Update & Free BMesh
    bm.to_mesh(cylinder.data)
    bm.free()

    if U() < 0.5:
        cylinder.rotation_euler = (math.pi / 2, 0, 0)
    else:
        cylinder.rotation_euler = (-math.pi / 2, 0, 0)
    bpy.context.view_layer.objects.active = cylinder
    cylinder.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    return cylinder

def sample_plane_location(camera, moving_threshold, factor_distance=0.5, factor_size=0.5):
    distance = -U(moving_threshold * factor_distance, moving_threshold)
    focal_length = camera.data.lens
    sensor_width = camera.data.sensor_width
    sensor_height = camera.data.sensor_height
    horizontal_fov = 2 * math.atan(sensor_width / (2 * focal_length))
    vertical_fov = 2 * math.atan(sensor_height / (2 * focal_length))
    canvas_width = 2 * math.tan(horizontal_fov / 2) * abs(distance)
    canvas_height = 2 * math.tan(vertical_fov / 2) * abs(distance)
    canvas_length = min(canvas_width, canvas_height)

    size = N(factor_size * canvas_length, 0.1 * factor_size * canvas_length)
    x_shift = N(0, 0.2 * canvas_width)
    y_shift = N(0, 0.2 * canvas_height)
    location = camera.location + camera.rotation_euler.to_matrix() @ Vector((x_shift, y_shift, distance))
    return location, size

def check_valid_location(scene, location):
    for obj in scene.objects:
        if obj.type == 'MESH':
            bounding_box = obj.bound_box
            min_coords = [min([v[i] for v in bounding_box]) for i in range(3)]
            max_coords = [max([v[i] for v in bounding_box]) for i in range(3)]
            if min_coords[0] < location[0] < max_coords[0] \
                and min_coords[1] < location[1] < max_coords[1] \
                and min_coords[2] < location[2] < max_coords[2]:
                return False            
    return True

def add_flying_objects(scene, args, config, use_asset=None, counter=0, add_objects=True):
    planes = []
    if "moving_threshold" in config:
        moving_threshold = config["moving_threshold"]
    else:
        moving_threshold = [u - v for u, v in zip(config["camera_range"]["loc_upper_bound"], config["camera_range"]["loc_lower_bound"])]
        moving_threshold = sum(moving_threshold) / len(moving_threshold)

    plane_number = max(int(N(1, 1)), 1)

    for i in range(plane_number):
        camera = scene.camera

        trys = 10
        location = None
        while trys > 0:
            location, size = sample_plane_location(camera, moving_threshold, 0.3, 0.4)
            if check_valid_location(scene, location):
                break
            trys -= 1
        
        if location is None:
            continue
        
        thickness = U(0.02, 0.05)
        rotation_euler = camera.rotation_euler.copy()
        rotation_euler = perturb_rotation_matrix(rotation_euler)
        if use_asset is None:
            use_asset = U() < 0.2

        if use_asset:
            obj = get_object(counter, 'window_assets')
            obj.dimensions = Vector((size + N(0, 0.1), size + N(0, 0.1), U(0, 0.1)))
            add_randomness_to_glass(obj)
            set_object_location_and_rotation(obj, location, rotation_euler, moving_threshold)
            planes.append(obj)
        else:
            if U() < 0.85:
                bpy.ops.mesh.primitive_plane_add(size=size, location=camera.location)
                plane = bpy.context.object
            else:
                plane = get_cylinder_plane(size)
            solidify = plane.modifiers.new(name="Solidify", type='SOLIDIFY')
            solidify.thickness = thickness
            set_object_location_and_rotation(plane, location, rotation_euler, moving_threshold)
            assign_glass_material(plane, ior=1.45)
            planes.append(plane)
    
    if add_objects:
        object_number = max(int(N(4, 1)), 0)
        for counter in range(object_number):
            camera = scene.camera

            trys = 10
            location = None
            while trys > 0:
                location, size = sample_plane_location(camera, moving_threshold, 0.7, 0.02)
                if check_valid_location(scene, location):
                    break
                trys -= 1
            
            if location is None:
                continue

            obj = get_object(counter, 'kitchen_assets')
            if obj is None:
                continue
            material_number = U()
            if material_number < args.glass_prob:
                if "clearable" in obj.name or "cleanable" in obj.name:
                    assign_glass_material(obj, ior=1.45)
                else:
                    assign_rough_glass_material(obj)
            elif material_number < args.glass_prob + args.metal_prob:
                assign_metalic_material(obj)
            
            random_angle_x = U(0, math.pi * 2)
            random_angle_y = U(0, math.pi * 2)
            random_angle_z = U(0, math.pi * 2)
            rotation_euler = Vector((random_angle_x, random_angle_y, random_angle_z))
            set_object_location_and_rotation(obj, location, rotation_euler, moving_threshold)

            sqrt_dim = (obj.dimensions[0] ** 2 + obj.dimensions[1] ** 2 + obj.dimensions[2] ** 2) ** 0.5

            relative_scale = size / sqrt_dim
            random_scale_x = relative_scale
            random_scale_y = N(random_scale_x, 0.01) 
            random_scale_z = N(random_scale_x, 0.01)
            obj.scale[0] = random_scale_x
            obj.scale[1] = random_scale_y
            obj.scale[2] = random_scale_z
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            planes.append(obj)

    return planes

def adjust_object_visibility(obj, visibility):
    if obj is None:
        return
    elif type(obj) == list:
        for o in obj:
            adjust_object_visibility(o, visibility)
        return
    else:
        obj.hide_render = not visibility
        for child in obj.children:
            child.hide_render = not visibility

def random_objects(args, config):
    movable_number = len([obj for obj in bpy.data.objects if "movable" in obj.name])
    if movable_number > 10:
        args.random_place = 5 / movable_number
        ## override random place for too many objects scene

    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            if obj.name.startswith("border"):
                continue
            if is_obj_has_emission_material(obj):
                continue

            ## Assign material
            material_number = U()
            if material_number < args.glass_prob:
                if "clearable" in obj.name or "cleanable" in obj.name:
                    assign_glass_material(obj, ior=1.45)
                else:
                    assign_rough_glass_material(obj)
            elif material_number < args.glass_prob + args.metal_prob:
                assign_metalic_material(obj)


