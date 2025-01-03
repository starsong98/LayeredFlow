# Adopted from https://github.com/Cartucho/vision_blender
import json
import os
import shutil
import numpy as np
import cv2
import bpy
import time
from .utils import save_img

def get_scene_resolution(scene):
    resolution_scale = (scene.render.resolution_percentage / 100.0)
    resolution_x = scene.render.resolution_x * resolution_scale # [pixels]
    resolution_y = scene.render.resolution_y * resolution_scale # [pixels]
    return int(resolution_x), int(resolution_y)

def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit

def get_camera_parameters_intrinsic(scene):
    """ Get intrinsic camera parameters: focal length and principal point. """
    # ref: https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera/120063#120063
    focal_length = scene.camera.data.lens # [mm]
    res_x, res_y = get_scene_resolution(scene)
    cam_data = scene.camera.data
    sensor_size_in_mm = get_sensor_size(cam_data.sensor_fit, cam_data.sensor_width, cam_data.sensor_height)
    sensor_fit = get_sensor_fit(
        cam_data.sensor_fit,
        scene.render.pixel_aspect_x * res_x,
        scene.render.pixel_aspect_y * res_y
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = res_x
    else:
        view_fac_in_px = pixel_aspect_ratio * res_y
    pixel_size_mm_per_px = (sensor_size_in_mm / focal_length) / view_fac_in_px
    f_x = 1.0 / pixel_size_mm_per_px
    f_y = (1.0 / pixel_size_mm_per_px) / pixel_aspect_ratio
    c_x = (res_x - 1) / 2.0 - cam_data.shift_x * view_fac_in_px
    c_y = (res_y - 1) / 2.0 + (cam_data.shift_y * view_fac_in_px) / pixel_aspect_ratio
    return f_x, f_y, c_x, c_y

def get_camera_parameters_extrinsic(scene):
    """ Get extrinsic camera parameters. 
    
      There are 3 coordinate systems involved:
         1. The World coordinates: "world"
            - right-handed
         2. The Blender camera coordinates: "bcam"
            - x is horizontal
            - y is up
            - right-handed: negative z look-at direction
         3. The desired computer vision camera coordinates: "cv"
            - x is horizontal
            - y is down (to align to the actual pixel coordinates 
               used in digital images)
            - right-handed: positive z look-at direction

      ref: https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
    """
    # bcam stands for blender camera
    bcam = scene.camera
    R_bcam2cv = np.array([[1,  0,  0],
                          [0, -1,  0],
                          [0,  0, -1]])

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location = np.array([bcam.matrix_world.decompose()[0]]).T
    R_world2bcam = np.array(bcam.matrix_world.decompose()[1].to_matrix().transposed())

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*bcam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = np.matmul(R_world2bcam.dot(-1), location)

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = np.matmul(R_bcam2cv, R_world2bcam)
    T_world2cv = np.matmul(R_bcam2cv, T_world2bcam)

    extr = np.concatenate((R_world2cv, T_world2cv), axis=1)
    return extr

def get_obj_poses():
    n_chars = get_largest_object_name_length()
    n_object = len(bpy.data.objects)
    obj_poses = np.zeros(n_object, dtype=[('name', 'U{}'.format(n_chars)), ('pose', np.float64, (4, 4))])
    for ind, obj in enumerate(bpy.data.objects):
        obj_poses[ind] = (obj.name, obj.matrix_world)
    return obj_poses

def check_if_node_exists(tree, node_name):
    node_ind = tree.nodes.find(node_name)
    if node_ind == -1:
        return False
    return True

def create_node(tree, node_type, node_name):
    node_exists = check_if_node_exists(tree, node_name)
    if not node_exists:
        v = tree.nodes.new(node_type)
        v.name = node_name
    else:
        v = tree.nodes[node_name]
    return v

def remove_old_vision_blender_nodes(tree):
    for node in tree.nodes:
        if 'vision_blender' in node.name:
            tree.nodes.remove(node)

def clean_folder(folder_path):
    # ref : https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        os.makedirs(folder_path)

def get_set_of_non_zero_obj_ind():
    non_zero_obj_ind_set = set([])
    for obj in bpy.data.objects:
        if obj.pass_index != 0:
            non_zero_obj_ind_set.add(obj.pass_index)
    return non_zero_obj_ind_set

def check_any_obj_with_non_zero_index():
    for obj in bpy.data.objects:
        if obj.pass_index != 0:
            return True
    return False

def check_any_material_with_non_zero_index():
    for obj in bpy.data.objects:
        for i, material_slot in enumerate(obj.material_slots):
            if material_slot.material.pass_index != 0:
                return True
    return False

def get_largest_object_name_length():
    max_chars = 0
    for obj in bpy.data.objects:
        if len(obj.name) > max_chars:
            max_chars = len(obj.name)
    return max_chars

def get_struct_array_of_obj_indexes():
    # ref: https://numpy.org/doc/stable/user/basics.rec.html
    n_chars = get_largest_object_name_length()
    n_object = len(bpy.data.objects)
    # max_index = 32767 in Blender version 2.83, so unsigned 2-byte is more than enough memory
    obj_indexes = np.zeros(n_object, dtype=[('name', 'U{}'.format(n_chars)), ('pass_index', '<u2')])
    for ind, obj in enumerate(bpy.data.objects):
        obj_indexes[ind] = (obj.name, obj.pass_index)
    return obj_indexes

def is_stereo_ok_for_disparity(scene):
    if (scene.render.use_multiview and
        scene.camera.data.stereo.convergence_mode == 'PARALLEL'):
        return True
    return False

def get_transf0to1(scene):
    transf = None
    if scene.camera.data.stereo.convergence_mode == 'PARALLEL':
        translation_x = scene.camera.data.stereo.interocular_distance
        transf = np.zeros((4, 4))
        np.fill_diagonal(transf, 1)
        transf[0, 3] = - translation_x
    return transf

def load_file_data_to_numpy(scene, tmp_file_path, data_map):
    if not os.path.isfile(tmp_file_path):
        return None
    out_data = bpy.data.images.load(tmp_file_path)
    pixels_numpy = np.array(out_data.pixels[:])
    res_x, res_y = get_scene_resolution(scene)
    pixels_numpy.resize((res_y, res_x, 4)) # Numpy works with (y, x, channels)
    pixels_numpy = np.flip(pixels_numpy, 0) # flip vertically (in Blender y in the image points up instead of down)
    if data_map == 'Normal' or data_map.startswith('Denoising_Position') or data_map.startswith('Denoising_Normal'): # 3 Channels
        result = pixels_numpy[:, :, 0:3]
        return result
    
    elif data_map == 'Depth': # 1 Channel
        z = pixels_numpy[:, :, 0]
        max_dist = scene.camera.data.clip_end
        INVALID_POINT = -1.0
        z[z > max_dist] = INVALID_POINT

        # If stereo also calculate disparity
        disp = None
        if not is_stereo_ok_for_disparity(scene):
            return z, disp
        
        baseline_m = scene.camera.data.stereo.interocular_distance # [m]
        disp = np.zeros_like(z) # disp = 0.0, on the invalid points
        f_x, _f_y, _c_x, _c_y = get_camera_parameters_intrinsic(scene) # needed for z in Cycles and for disparity
        disp[z != INVALID_POINT] = (baseline_m * f_x) / (z[z != INVALID_POINT] + 1e-6)
        # Check `tmp_file_path` if it is for the left or right camera
        suffix1 = scene.render.views[1].file_suffix
        if suffix1 in tmp_file_path: # By default, if '_R' in `tmp_file_path`
            np.negative(disp)
        return z, disp
    
    elif data_map == 'Segmentation' or data_map.startswith('Denoising_Mask'):
        tmp_seg_mask = pixels_numpy[:,:,0]
        return tmp_seg_mask
    
    elif data_map == 'OptFlow' or data_map.startswith('Denoising_Vector'):
        opt_flw = pixels_numpy[:,:,:2] 
        opt_flw[:,:,1] = np.negative(opt_flw[:,:,1]) 
        return opt_flw

def save_data_to_npz(scene, is_stereo_activated, results, obj_poses):
    # ref: https://stackoverflow.com/questions/35133317/numpy-save-some-arrays-at-once
    
    def save_results(result, id):
        image_path = os.path.dirname(scene.render.filepath)
        save_img(result, image_path, id)

        suffix = scene.render.views[id].file_suffix
        data_path = os.path.join(image_path, f'data{suffix}.npz')
        result_filtered = {k: v for k, v in result.items() if v is not None}
        np.savez_compressed(data_path, **result_filtered)

    cam0_result = results['cam0']
    save_results(cam0_result, 0)
    if is_stereo_activated:
        cam1_result = results['cam1']
        save_results(cam1_result, 1)

def gt_init(scene, coef, gts=[], no_rgb=False):
    """ This function is called before starting to render """
    # check if user wants to generate the ground truth data

    # 1. Set-up Passes
    if not scene.use_nodes:
        scene.use_nodes = True
    if 'DEPTH' in gts:
        if not bpy.context.view_layer.use_pass_z:
            bpy.context.view_layer.use_pass_z = True
    if 'NORMAL' in gts:
        if not bpy.context.view_layer.use_pass_normal:
            bpy.context.view_layer.use_pass_normal = True
    if 'DENOISE' in gts:
        bpy.context.view_layer.cycles.denoising_store_passes = True
    
    ## Segmentation masks and optical flow only work in Cycles
    if scene.render.engine == 'CYCLES':
        if 'SEGMENT' in gts:
            if not bpy.context.view_layer.use_pass_object_index:
                bpy.context.view_layer.use_pass_object_index = True
            if not bpy.context.view_layer.use_pass_material_index:
                bpy.context.view_layer.use_pass_material_index = True
        if 'FLOW' in gts:
            if not bpy.context.view_layer.use_pass_vector:
                bpy.context.view_layer.use_pass_vector = True

    # """ All the data will be saved to a MultiLayer OpenEXR image. """
    # 2. Set-up nodes
    tree = scene.node_tree
    ## Remove old nodes (from previous rendering)
    remove_old_vision_blender_nodes(tree)
    ## Create new output node
    node_output = create_node(tree, "CompositorNodeOutputFile", "output_vision_blender")
    ## Set-up the output img format
    node_output.format.file_format = 'OPEN_EXR'
    node_output.format.color_mode = 'RGBA'
    node_output.format.color_depth = '32'
    node_output.format.exr_codec = 'PIZ'
    ## Set-up output path

    TMP_FILES_PATH = os.path.join(os.path.dirname(scene.render.filepath), 'tmp_vision_blender')
    # clean_folder(TMP_FILES_PATH)
    node_output.base_path = TMP_FILES_PATH

    # 3. Set-up links between nodes
    node_output.layer_slots.clear() # Remove all the default layer slots
    links = tree.links

    if "Render Layers" in tree.nodes:
        rl = scene.node_tree.nodes["Render Layers"] # I assumed there is always a Render Layers
    else:
        rl = tree.nodes.new('CompositorNodeRLayers')

    """ Normal map """
    if 'NORMAL' in gts:
        slot_normal = node_output.layer_slots.new('####_Normal')
        links.new(rl.outputs["Normal"], slot_normal)
    """ Depth map """
    if 'DEPTH' in gts:
        slot_depth = node_output.layer_slots.new('####_Depth')
        links.new(rl.outputs["Depth"], slot_depth)

    # 4. Set-up nodes and links for Cycles only (for optical flow and segmentation masks)
    if scene.render.engine == "CYCLES":
        """ Denoising data """
        if 'DENOISE' in gts:
            slot_denoise_position = {}
            slot_denoise_normal = {}
            slot_denoise_forward_vector = {}
            slot_denoise_backward_vector = {}
            slot_denoise_mask = {}

            for i in range(4):
                slot_denoise_position[i] = node_output.layer_slots.new(f'####_Denoising_Position_{i}')
                links.new(rl.outputs[f"Denoising Position {i}"], slot_denoise_position[i])

                slot_denoise_forward_vector[i] = node_output.layer_slots.new(f'####_Denoising_Forward_Vector_{i}')
                slot_denoise_backward_vector[i] = node_output.layer_slots.new(f'####_Denoising_Backward_Vector_{i}')
                node_separate = create_node(tree, "CompositorNodeSepRGBA", f"sep_{i}_vision_blender")
                node_combine_forward = create_node(tree, "CompositorNodeCombRGBA", f"comb_forward_{i}_vision_blender")
                node_combine_backward = create_node(tree, "CompositorNodeCombRGBA", f"comb_backward_{i}_vision_blender")
                links.new(rl.outputs[f"Denoising Vector {i}"], node_separate.inputs["Image"])
                links.new(node_separate.outputs["R"], node_combine_backward.inputs["R"])
                links.new(node_separate.outputs["G"], node_combine_backward.inputs["G"])
                links.new(node_separate.outputs["B"], node_combine_forward.inputs["R"])
                links.new(node_separate.outputs["A"], node_combine_forward.inputs["G"])
                links.new(node_combine_forward.outputs['Image'], slot_denoise_forward_vector[i])
                links.new(node_combine_backward.outputs['Image'], slot_denoise_backward_vector[i])

                slot_denoise_mask[i] = node_output.layer_slots.new(f'####_Denoising_Mask_{i}')
                links.new(rl.outputs[f"Denoising Mask {i}"], slot_denoise_mask[i])
            
        """ Segmentation masks """
        if 'SEGMENT' in gts:
            # We can only generate segmentation masks if that are any labeled objects (objects w/ index set)
            non_zero_obj_ind_found = check_any_material_with_non_zero_index()
            if non_zero_obj_ind_found:
                slot_seg_mask = node_output.layer_slots.new('####_Segmentation_Mask')
                links.new(rl.outputs["IndexMA"], slot_seg_mask)
        """ Optical flow - Current to next frame """
        if 'FLOW' in gts:
            # Create new slot in output node
            slot_opt_flow = node_output.layer_slots.new("####_Optical_Flow")
            # Get optical flow
            node_rg_separate = create_node(tree, "CompositorNodeSepRGBA", "BA_sep_vision_blender")
            node_rg_combine = create_node(tree, "CompositorNodeCombRGBA", "RG_comb_vision_blender")
            links.new(rl.outputs["Vector"], node_rg_separate.inputs["Image"])
            links.new(node_rg_separate.outputs["B"], node_rg_combine.inputs["R"])
            links.new(node_rg_separate.outputs["A"], node_rg_combine.inputs["G"])
            # Connect to output node
            links.new(node_rg_combine.outputs['Image'], slot_opt_flow)

    # 5. Save camera_info (for vision_blender_ros)
    camera_location, camera_direction, lens = coef
    dict_cam_info = {}
    render = scene.render
    cam = scene.camera
    dict_cam_info['location'] = list(camera_location)
    dict_cam_info['direction'] = list(camera_direction)
    dict_cam_info['lens'] = lens
    ## img file format
    dict_cam_info['img_format'] = render.image_settings.file_format
    ## camera image resolution
    res_x, res_y = get_scene_resolution(scene)
    dict_cam_info['img_res_x'] = res_x
    dict_cam_info['img_res_y'] = res_y
    ## camera intrinsic matrix parameters
    cam_mat_intr = {}
    f_x, f_y, c_x, c_y = get_camera_parameters_intrinsic(scene)
    cam_mat_intr['f_x'] = f_x
    cam_mat_intr['f_y'] = f_y
    cam_mat_intr['c_x'] = c_x
    cam_mat_intr['c_y'] = c_y
    dict_cam_info['cam_mat_intr'] = cam_mat_intr
    ## is_stereo
    is_stereo = render.use_multiview
    dict_cam_info['is_stereo'] = is_stereo
    if is_stereo:
        stereo_info = {}
        ### left camera file suffix
        stereo_info['stereo_left_suffix'] = render.views["left"].file_suffix
        ### right camera file suffix
        stereo_info['stereo_right_suffix'] = render.views["right"].file_suffix
        ### stereo mode
        stereo_info['stereo_mode'] = cam.data.stereo.convergence_mode
        ### stereo interocular distance
        stereo_info['stereo_interocular_distance [m]'] = cam.data.stereo.interocular_distance
        ### stereo pivot
        stereo_info['stereo_pivot'] = cam.data.stereo.pivot
        dict_cam_info['stereo_info'] = stereo_info
    ## save data to a json file
    gt_dir_path = os.path.dirname(scene.render.filepath)
    out_path = os.path.join(gt_dir_path, 'camera_info.json')
    with open(out_path, 'w') as tmp_file:
        json.dump(dict_cam_info, tmp_file, indent=4, sort_keys=True)

def gt_after_frame(scene, gts=[]): 
    """ This script runs after rendering each frame """
    # ref: https://blenderartists.org/t/how-to-run-script-on-every-frame-in-blender-render/699404/2
    # check if user wants to generate the ground truth data

    is_stereo_activated = scene.render.use_multiview
    if is_stereo_activated:
        suffix0 = scene.render.views[0].file_suffix # By default '_L'
        suffix1 = scene.render.views[1].file_suffix # By default '_R'
    
    """ Camera parameters """
    ## update camera - ref: https://blender.stackexchange.com/questions/5636/how-can-i-get-the-location-of-an-object-at-each-keyframe

    scene.frame_set(scene.frame_current) # needed to update the camera position
    intrinsic_mat = None
    extrinsic_mat0 = None
    extrinsic_mat1 = None
    extrinsic_mat0 = get_camera_parameters_extrinsic(scene)
 
    transf0to1 = get_transf0to1(scene)
    if transf0to1 is not None:
        extrinsic_mat0 = np.vstack((extrinsic_mat0, [0, 0, 0, 1.]))
        extrinsic_mat1 = np.matmul(transf0to1, extrinsic_mat0)
        # Remove homogeneous row
        extrinsic_mat0 = extrinsic_mat0[:3,:]
        extrinsic_mat1 = extrinsic_mat1[:3,:]
    # Intrinsic mat is the same for both stereo cameras
    f_x, f_y, c_x, c_y = get_camera_parameters_intrinsic(scene)
    intrinsic_mat = np.array([[f_x,   0,  c_x],
                                [  0, f_y,  c_y],
                                [  0,   0,    1]])
    """ Objects' pose """
    obj_poses = None
    # obj_poses = get_obj_poses()
    
    """ Get the data from the output node """
    normal0 = None
    normal1 = None
    z0 = None
    z1 = None
    disp0 = None
    disp1 = None
    de_position_0 = []
    de_position_1 = []
    de_normal_0 = []
    de_normal_1 = []
    de_forward_vector_0 = []
    de_forward_vector_1 = []
    de_backward_vector_0 = []
    de_backward_vector_1 = []
    de_vector_0 = []
    de_vector_1 = []
    de_mask_0 = []
    de_mask_1 = []
    seg_masks0 = None
    seg_masks1 = None
    seg_masks_indexes = None
    opt_flw0 = None
    opt_flw1 = None

    if check_if_node_exists(scene.node_tree, 'output_vision_blender'):
        node_output = scene.node_tree.nodes['output_vision_blender']
        TMP_FILES_PATH = node_output.base_path
        """ Normal map """
        if 'NORMAL' in gts:
            if is_stereo_activated:
                tmp_file_path1 = os.path.join(TMP_FILES_PATH, '{:04d}_Normal{}.exr'.format(scene.frame_current, suffix1))
                normal1 = load_file_data_to_numpy(scene, tmp_file_path1, 'Normal')
                tmp_file_path0 = os.path.join(TMP_FILES_PATH, '{:04d}_Normal{}.exr'.format(scene.frame_current, suffix0))
            else:
                tmp_file_path0 = os.path.join(TMP_FILES_PATH, '{:04d}_Normal.exr'.format(scene.frame_current))
            normal0 = load_file_data_to_numpy(scene, tmp_file_path0, 'Normal')
        """ Depth + Disparity """
        if 'DEPTH' in gts:
            if is_stereo_activated:
                tmp_file_path1 = os.path.join(TMP_FILES_PATH, '{:04d}_Depth{}.exr'.format(scene.frame_current, suffix1))
                z1, disp1 = load_file_data_to_numpy(scene, tmp_file_path1, 'Depth')
                tmp_file_path0 = os.path.join(TMP_FILES_PATH, '{:04d}_Depth{}.exr'.format(scene.frame_current, suffix0))
            else:
                tmp_file_path0 = os.path.join(TMP_FILES_PATH, '{:04d}_Depth.exr'.format(scene.frame_current))
            z0, disp0 = load_file_data_to_numpy(scene, tmp_file_path0, 'Depth')

        if scene.render.engine == "CYCLES":
            """ Denoising data """
            if 'DENOISE' in gts:
                for layer in range(4):
                    if is_stereo_activated:
                        tmp_file_path1 = os.path.join(TMP_FILES_PATH, '{:04d}_Denoising_Position_{}{}.exr'.format(scene.frame_current, layer, suffix1))
                        de_position_1.append(load_file_data_to_numpy(scene, tmp_file_path1, f'Denoising_Position_{layer}'))
                        tmp_file_path0 = os.path.join(TMP_FILES_PATH, '{:04d}_Denoising_Position_{}{}.exr'.format(scene.frame_current, layer, suffix0))
                    else:
                        tmp_file_path0 = os.path.join(TMP_FILES_PATH, '{:04d}_Denoising_Position_{}.exr'.format(scene.frame_current, layer))
                    de_position_0.append(load_file_data_to_numpy(scene, tmp_file_path0, f'Denoising_Position_{layer}'))

                    if is_stereo_activated:
                        tmp_file_path1 = os.path.join(TMP_FILES_PATH, '{:04d}_Denoising_Forward_Vector_{}{}.exr'.format(scene.frame_current, layer, suffix1))
                        de_forward_vector_1.append(load_file_data_to_numpy(scene, tmp_file_path1, f'Denoising_Vector_{layer}'))
                        tmp_file_path0 = os.path.join(TMP_FILES_PATH, '{:04d}_Denoising_Forward_Vector_{}{}.exr'.format(scene.frame_current, layer, suffix0))
                    else:
                        tmp_file_path0 = os.path.join(TMP_FILES_PATH, '{:04d}_Denoising_Forward_Vector_{}.exr'.format(scene.frame_current, layer))
                    de_forward_vector_0.append(load_file_data_to_numpy(scene, tmp_file_path0, f'Denoising_Vector_{layer}'))

                    if is_stereo_activated:
                        tmp_file_path1 = os.path.join(TMP_FILES_PATH, '{:04d}_Denoising_Backward_Vector_{}{}.exr'.format(scene.frame_current, layer, suffix1))
                        de_backward_vector_1.append(load_file_data_to_numpy(scene, tmp_file_path1, f'Denoising_Vector_{layer}'))
                        tmp_file_path0 = os.path.join(TMP_FILES_PATH, '{:04d}_Denoising_Backward_Vector_{}{}.exr'.format(scene.frame_current, layer, suffix0))
                    else:
                        tmp_file_path0 = os.path.join(TMP_FILES_PATH, '{:04d}_Denoising_Backward_Vector_{}.exr'.format(scene.frame_current, layer))
                    de_backward_vector_0.append(load_file_data_to_numpy(scene, tmp_file_path0, f'Denoising_Vector_{layer}'))
                    
                    if is_stereo_activated:
                        tmp_file_path1 = os.path.join(TMP_FILES_PATH, '{:04d}_Denoising_Mask_{}{}.exr'.format(scene.frame_current, layer, suffix1))
                        de_mask_1.append(load_file_data_to_numpy(scene, tmp_file_path1, f'Denoising_Mask_{layer}'))
                        tmp_file_path0 = os.path.join(TMP_FILES_PATH, '{:04d}_Denoising_Mask_{}{}.exr'.format(scene.frame_current, layer, suffix0))
                    else:
                        tmp_file_path0 = os.path.join(TMP_FILES_PATH, '{:04d}_Denoising_Mask_{}.exr'.format(scene.frame_current, layer))
                    de_mask_0.append(load_file_data_to_numpy(scene, tmp_file_path0, f'Denoising_Mask_{layer}'))
                
                for forward, backward in zip(de_forward_vector_0, de_backward_vector_0):
                    de_vector_0.append(np.concatenate((backward, forward), axis=-1))
                if is_stereo_activated:
                    for forward, backward in zip(de_forward_vector_1, de_backward_vector_1):
                        de_vector_1.append(np.concatenate((backward, forward), axis=-1))

            """ Segmentation masks """
            if 'SEGMENT' in gts and check_any_material_with_non_zero_index():
                # seg_masks_indexes = get_struct_array_of_obj_indexes()
                seg_masks_indexes = {1: 1}
                if is_stereo_activated:
                    tmp_file_path1 = os.path.join(TMP_FILES_PATH, '{:04d}_Segmentation_Mask{}.exr'.format(scene.frame_current, suffix1))
                    seg_masks1 = load_file_data_to_numpy(scene, tmp_file_path1, 'Segmentation')
                    tmp_file_path0 = os.path.join(TMP_FILES_PATH, '{:04d}_Segmentation_Mask{}.exr'.format(scene.frame_current, suffix0))
                else:
                    tmp_file_path0 = os.path.join(TMP_FILES_PATH, '{:04d}_Segmentation_Mask.exr'.format(scene.frame_current))
                seg_masks0 = load_file_data_to_numpy(scene, tmp_file_path0, 'Segmentation')
            """ Optical flow - Forward -> from current to next frame"""
            if 'FLOW' in gts:
                if is_stereo_activated:
                    tmp_file_path1 = os.path.join(TMP_FILES_PATH, '{:04d}_Optical_Flow{}.exr'.format(scene.frame_current, suffix1))
                    opt_flw1 = load_file_data_to_numpy(scene, tmp_file_path1, 'OptFlow')
                    tmp_file_path0 = os.path.join(TMP_FILES_PATH, '{:04d}_Optical_Flow{}.exr'.format(scene.frame_current, suffix0))
                else:
                    tmp_file_path0 = os.path.join(TMP_FILES_PATH, '{:04d}_Optical_Flow.exr'.format(scene.frame_current))
                opt_flw0 = load_file_data_to_numpy(scene, tmp_file_path0, 'OptFlow')

        results = {}
        results['cam0'] = {
            'optical_flow': opt_flw0,
            'segmentation_masks': seg_masks0,
            'segmentation_masks_indexes': seg_masks_indexes,
            'intrinsic_mat': intrinsic_mat,
            'extrinsic_mat0': extrinsic_mat0,
            'normal_map': normal0,
            'depth_map': z0,
            'disparity_map': disp0,
            'denoising_position_map': de_position_0,
            'denoising_normal_map': de_normal_0,
            'denoising_vector_map': de_vector_0,
            'denoising_mask_map': de_mask_0,
            'object_poses': obj_poses,
        }
        if is_stereo_activated:
            results['cam1'] = {
                'optical_flow': opt_flw1,
                'segmentation_masks': seg_masks1,
                'segmentation_masks_indexes': seg_masks_indexes,
                'intrinsic_mat': intrinsic_mat,
                'extrinsic_mat1': extrinsic_mat1,
                'normal_map': normal1,
                'depth_map': z1,
                'disparity_map': disp1,
                'denoising_position_map': de_position_1,
                'denoising_normal_map': de_normal_1,
                'denoising_vector_map': de_vector_1,
                'denoising_mask_map': de_mask_1,
                'object_poses': obj_poses,
            }

        # Optional step - delete the tmp output files
        # clean_folder(TMP_FILES_PATH)
        """ Save data """
        save_data_to_npz(scene, is_stereo_activated, results, obj_poses)

def gt_finish(scene):
    if check_if_node_exists(scene.node_tree, 'output_vision_blender'):
        node_output = scene.node_tree.nodes['output_vision_blender']
        TMP_FILES_PATH = node_output.base_path
        shutil.rmtree(TMP_FILES_PATH)
