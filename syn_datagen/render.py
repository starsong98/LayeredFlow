import bpy
import bmesh
import json
import argparse
import random
import math
import cv2
import string
import shutil
from mathutils import Vector
import sys, os
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from numpy.random import uniform as U, normal as N, randint as RI

sys.path.append(os.path.join(os.path.dirname(__file__)))
from utils.utils import stdout_redirected
from utils.material import set_color_to_all, emission_invisible
from utils.camera import remove_all_camera, get_camera_sample, setup_stereo_camera
from utils.blender_gt import gt_init, gt_after_frame, gt_finish
from utils.geom import add_flying_objects, adjust_object_visibility, random_objects

def set_hdr(hdr_path):
    world = bpy.context.scene.world

    env_strength = 1.0
    if world.use_nodes == True:
        node_tree = world.node_tree
        for node in node_tree.nodes:
            if isinstance(node, bpy.types.ShaderNodeBackground):
                env_strength = node.inputs["Strength"].default_value
    
    world.use_nodes = True
    world.node_tree.nodes.clear()

    node_tree = world.node_tree
    tex_coord_node = node_tree.nodes.new('ShaderNodeTexCoord')

    mapping_node = node_tree.nodes.new('ShaderNodeMapping')
    # mapping_node.inputs['Rotation'].default_value[2] = 0.5
    node_tree.links.new(tex_coord_node.outputs['Generated'], mapping_node.inputs['Vector'])

    texture_node = node_tree.nodes.new('ShaderNodeTexEnvironment')
    texture_node.image = bpy.data.images.load(filepath=hdr_path)
    node_tree.links.new(mapping_node.outputs['Vector'], texture_node.inputs['Vector'])

    background_node = node_tree.nodes.new('ShaderNodeBackground')
    background_node.inputs['Strength'].default_value = env_strength
    node_tree.links.new(texture_node.outputs['Color'], background_node.inputs['Color'])
    output_node = node_tree.nodes.new('ShaderNodeOutputWorld')
    node_tree.links.new(background_node.outputs['Background'], output_node.inputs['Surface'])

def remove_tmp_files(path):
    for file in os.listdir(path):
        if file.startswith("tmp"):
            os.remove(os.path.join(path, file))

def setup_scene(args, config, scene):
    if args.test:
        args.resolution = (128, 72)
        args.samples = 16
        args.output_folder = 'images_test'
        args.plane_prob = 0.7
        args.metal_prob = 0.3
        args.glass_prob = 0.4
        args.random_place = 0.5
    if args.data:
        args.resolution = (1280, 720)
        args.samples = 1024
        args.output_folder = 'images_data'
        args.plane_prob = 0.7
        args.metal_prob = 0.3
        args.glass_prob = 0.4
        args.random_place = 0.5
    if args.sample:
        args.sample_view = True
        args.resolution = (1280, 720)
        args.samples = 1024
        args.num_images = 1
        args.output_folder = 'images_sample'
        args.plane_prob = 0.0
        args.metal_prob = 0.0
        args.glass_prob = 0.0
    
    # setup hdr
    hdr_number = U()
    if hdr_number < args.random_hdr:
        hdrs = os.listdir("external_hdrs")
        hdr_file = random.choice(hdrs)
        scene.render.film_transparent = False
        set_hdr(os.path.join("external_hdrs", hdr_file))
        print("Using HDR", hdr_file)

    ## reduce noise & fireflies
    bpy.context.scene.cycles.filter_glossy = 1.0 
    view_layer = bpy.context.view_layer

    # setup denoising
    view_layer.cycles.use_denoising = True
    bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'

    # setup clamp to reduce fireflies
    if "clamp" in config:
        clamp = U(config["clamp"][0], config["clamp"][1])
        bpy.context.scene.cycles.sample_clamp_direct = 0
        bpy.context.scene.cycles.sample_clamp_indirect = clamp
    else:
        bpy.context.scene.cycles.sample_clamp_direct = 0
        bpy.context.scene.cycles.sample_clamp_indirect = 2.0 + U(1, 3)
    
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = args.samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.02
    scene.cycles.device = "GPU"
    scene.cycles.transparent_max_bounces = 24
    scene.cycles.use_persistent_data = True
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    scene.cycles.pixel_filter_type = 'BLACKMAN_HARRIS'
    scene.cycles.filter_width = 1.0
    scene.render.image_settings.file_format = 'PNG'
    scene.render.use_multiview = True
    scene.render.views_format = 'STEREO_3D'
    scene.render.resolution_x = args.resolution[0]
    scene.render.resolution_y = args.resolution[1]
    scene.render.views[0].file_suffix = '_0'
    scene.render.views[1].file_suffix = '_1'
    scene.render.use_motion_blur = False

    render = bpy.context.scene.render

    # Disable the metadata
    render.use_stamp_date = False
    render.use_stamp_time = False
    render.use_stamp_render_time = False
    render.use_stamp_frame = False
    render.use_stamp_scene = False
    render.use_stamp_camera = False
    render.use_stamp_lens = False
    render.use_stamp_filename = False
    render.use_stamp_marker = False
    render.use_stamp_sequencer_strip = False
    render.use_stamp_note = False
    render.use_stamp_frame_range = False
    render.use_stamp_memory = False
    render.use_stamp_hostname = False
    render.use_stamp_labels = False

    ## Emission invisible to camera
    emission_invisible(scene)

    if "light_adjust" in config:
        adjust_light(config["light_adjust"])

def get_random_color():
    random_color = (1-abs(N(0, 0.15)), 1-abs(N(0, 0.15)), 1-abs(N(0, 0.15)), 1)
    return random_color

def adjust_light(light_adjust):
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            obj.data.energy = obj.data.energy * light_adjust
            obj.data.color = get_random_color()[:3]

    world = bpy.context.scene.world
    if world and world.use_nodes:
        for node in world.node_tree.nodes:
            if isinstance(node, bpy.types.ShaderNodeBackground):
                node.inputs["Strength"].default_value *= light_adjust

    for mat in bpy.data.materials:
        if mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == 'EMISSION':
                    node.inputs["Strength"].default_value *= light_adjust
                    node.inputs["Color"].default_value = get_random_color()

def dump_files(args, folder_name):
    folders = os.listdir(args.output_folder)
    folders = [folder for folder in folders if folder.startswith(str(args.scene_id) + '_')]
    new_folder_name = f"{args.scene_id}_{len(folders)}"

    shutil.move(os.path.join(args.output_folder, folder_name), os.path.join(args.output_folder, new_folder_name))

    image1 = os.path.join(args.output_folder, new_folder_name, "frame0", "image_0.png")
    image2 = os.path.join(args.output_folder, new_folder_name, "frame1", "image_0.png")
    flow1 = os.path.join(args.output_folder, new_folder_name, "frame0", "denoising_vector_forward_0_0.png")
    flow2 = os.path.join(args.output_folder, new_folder_name, "frame1", "denoising_vector_backward_0_0.png")
    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)
    flow1 = cv2.imread(flow1)
    flow2 = cv2.imread(flow2)

    stereo1 = image1
    stereo2 = os.path.join(args.output_folder, new_folder_name, "frame0", "image_1.png")
    stereo2 = cv2.imread(stereo2)

    image_concat = np.concatenate([image1, image2], axis=0)
    cv2.imwrite(os.path.join(args.output_folder, new_folder_name, "image_concat.png"), image_concat)

    flow_concat = np.concatenate([flow1, flow2], axis=0)
    cv2.imwrite(os.path.join(args.output_folder, new_folder_name, "flow_concat.png"), flow_concat)

    stereo_concat = np.concatenate([stereo1, stereo2], axis=1)
    for i in range(0, stereo1.shape[0], 100):
        cv2.line(stereo_concat, (0, i), (stereo_concat.shape[1], i), (0, 0, 0), 1)
    cv2.imwrite(os.path.join(args.output_folder, new_folder_name, "stereo_concat.png"), stereo_concat)

    # path = os.path.join(args.output_folder, new_folder_name, "frame0")
    # for file in os.listdir(path):
    #     if file.startswith("denoising") or file.startswith("gt"):
    #         os.remove(os.path.join(path, file))
    # path = os.path.join(args.output_folder, new_folder_name, "frame1")
    # for file in os.listdir(path):
    #     if file.startswith("denoising") or file.startswith("gt"):
    #         os.remove(os.path.join(path, file))

def setup_gpu(args):
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    gpu_list = []
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        if d.type == "CUDA":
            gpu_list.append(d)

    print("Using ", end="")
    if args.gpu_id == -1:
        for d in gpu_list:
            d["use"] = 1
            print(f"{d.name}, ", end="")
    else:
        for d in gpu_list:
            d["use"] = 0
        gpu_list[args.gpu_id]["use"] = 1
        print(f"{gpu_list[args.gpu_id].name} ", end="")
    print("for rendering")

def render_frame(folder_name, scene, frame_id, frame_name, render_gt=False):
    scene.frame_set(0)
    output_filepath = os.path.join(args.output_folder, folder_name, f"frame{frame_id}", f"{frame_name}.png")
    scene.render.filepath = output_filepath
    with stdout_redirected():
        bpy.ops.render.render(write_still = True)
    if render_gt:
        gt_after_frame(scene, gts=gts)

def save_json_info(args, config, coef, folder_name):
    info = {
        "scene_id": args.scene_id,
        "resolution": args.resolution,
        "samples": args.samples,
        "plane_prob": args.plane_prob,
        "metal_prob": args.metal_prob,
        "glass_prob": args.glass_prob,
        "random_place": args.random_place,
        "config": config,
        "camera_location": list(coef[0]),
        "camera_direction": list(coef[1]),
        "camera_lens": coef[2],
    }
    json.dump(info, open(os.path.join(args.output_folder, folder_name, f"info.json"), 'w'), indent=4)

gts = ["DENOISE"]
def main(args):
    print(f'Generating {args.num_images} images for scene {args.scene_id}')
    config = json.load(open(f'scene_configs/{args.scene_id}.json', 'r'))

    blend_file_path = os.path.join("external_blends", config["path"])
    bpy.ops.wm.open_mainfile(filepath=blend_file_path) 
    scene = bpy.context.scene

    setup_gpu(args)
    setup_scene(args, config, scene)
    remove_all_camera(scene)
    random_objects(args, config)

    camera = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera)
    scene.collection.objects.link(camera)
    scene.camera = camera

    camera_coefs, next_camera_coefs = get_camera_sample(scene.render.resolution_x, scene.render.resolution_y, camera.data.sensor_width, camera.data.sensor_height, scene, args, config)
    cameras = [setup_stereo_camera(scene, coef, next_coef, config) for coef, next_coef in zip(camera_coefs, next_camera_coefs)]

    if args.no_render:
        return
    
    print("Rendering images")
    # Render images
    flying_objects = []
    folder_names = [''.join(random.choices(string.ascii_lowercase, k=10)) for coef in camera_coefs]
    for camera_index, (coef, next_coef) in enumerate(zip(camera_coefs, next_camera_coefs)):
        folder_name = folder_names[camera_index]
        os.makedirs(os.path.join(args.output_folder, folder_name, "frame0"), exist_ok=True)
        os.makedirs(os.path.join(args.output_folder, folder_name, "frame1"), exist_ok=True)
        scene.camera = cameras[camera_index]
        
        objects = None
        if U() < args.plane_prob:
            objects = add_flying_objects(scene, args, config, counter=camera_index)
        flying_objects.append(objects)

        render_frame(folder_name, scene, 0, "image")
        render_frame(folder_name, scene, 1, "image")

        adjust_object_visibility(objects, False)
        if not args.generate_gt:
            folder_name = folder_names[camera_index]
            save_json_info(args, config, coef, folder_name)
            dump_files(args, folder_name)
        
    if not args.generate_gt:
        return
    
    # Render GT
    print("Rendering GT images")
    set_color_to_all(scene)
    adjust_light(10)
    for camera_index, (coef, next_coef) in enumerate(zip(camera_coefs, next_camera_coefs)):
        folder_name = folder_names[camera_index]
        os.makedirs(os.path.join(args.output_folder, folder_name), exist_ok=True)
        output_filepath = os.path.join(args.output_folder, folder_name, "frame0", f"gt.png")
        scene.render.filepath = output_filepath

        objects = flying_objects[camera_index]
        adjust_object_visibility(objects, True)
        scene.camera = cameras[camera_index]

        gt_init(scene, coef, gts=gts)        
        render_frame(folder_name, scene, 0, "gt", render_gt=True)
        render_frame(folder_name, scene, 1, "gt", render_gt=True)
        gt_finish(scene)
        
        adjust_object_visibility(objects, False)
        save_json_info(args, config, coef, folder_name)
        dump_files(args, folder_name)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, nargs=2, default=(1280, 720))
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--scene_id', type=int, default=0)
    parser.add_argument('--num_images', type=int, default=4)
    parser.add_argument('--output_folder', type=str, default='images')
   
    parser.add_argument('--glass_prob', type=float, default=0.2)
    parser.add_argument('--metal_prob', type=float, default=0.2)
    parser.add_argument('--plane_prob', type=float, default=0.1)
    parser.add_argument('--random_place', type=float, default=0.8)
    parser.add_argument('--random_hdr', type=float, default=0.8)
    parser.add_argument('--counter', type=int, default=0)

    parser.add_argument('--sample_view', action='store_true')
    parser.add_argument('--generate_gt', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--data', action='store_true')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--no_render', action='store_true')
    parser.add_argument('--no_disp', action='store_true')

    parser.add_argument('--gpu_id', type=int, default=-1)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    main(args)
