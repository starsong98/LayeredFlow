import matplotlib.pyplot as plt
import numpy as np
import colorsys
import cv2
import os, sys
from contextlib import contextmanager
from .flow_viz import flow_to_image

@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different


def flow_to_jet(flow):
    h, w, _ = flow.shape
    # flow = flow.copy().astype(np.float32)
    # flow[...,0] = flow[...,0] - flow[...,0].min()
    # flow[...,1] = flow[...,1] - flow[...,1].min()

    # if flow[...,0].max() > 0:
    #     flow[...,0] = flow[...,0] / flow[...,0].max()
    # if flow[...,1].max() > 0:
    #     flow[...,1] = flow[...,1] / flow[...,1].max()

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    angle =  cv2.normalize(angle, None, 0, 180, cv2.NORM_MINMAX)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv = np.zeros((h, w, 3), dtype=np.float32)
    hsv[..., 0] = angle  # Convert radians to degrees, mapping [0,pi] to [0,180]
    hsv[..., 1] = 255  # Saturation, you may want to adjust this based on your specific cases
    hsv[..., 2] = magnitude  # Value/Brightness channel

    # Convert HSV to BGR (for OpenCV) or RGB (for matplotlib)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # bgr = np.zeros((h, w, 3), dtype=np.float32)
    # bgr[..., 0] = flow[..., 0]
    # bgr[..., 1] = flow[..., 1]
    # bgr *= 255
    return bgr

def depth_to_jet(depth, scale_vmin=1.0):
    valid = (depth > 1e-3) & (depth < 1e4)
    vmin = depth[valid].min() * scale_vmin
    vmax = depth[valid].max()
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    depth = cmap(norm(depth))
    depth[~valid] = 1
    return np.ascontiguousarray(depth[...,:3] * 255, dtype=np.uint8)

def mask_to_color(mask, color_seed=None):
    classes = np.unique(mask)
    color_map = {}
    for i, class_id in enumerate(classes):
        color_map[class_id] = (np.asarray(colorsys.hsv_to_rgb(i / len(classes), 0.9, 0.8))*255).astype(np.uint8)

    segmentation_map_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype='uint8')
    for class_id, color in color_map.items():
        segmentation_map_color[mask == class_id] = color

    return segmentation_map_color

def save_img(out_dict, gt_dir_path, stereo_camera_id):
    depth = out_dict['depth_map'] if 'depth_map' in out_dict else None
    if depth is not None:
        depth = depth_to_jet(depth)
        cv2.imwrite(os.path.join(gt_dir_path, f'depth_{stereo_camera_id}.png'), depth)

    disp = out_dict['disparity_map'] if 'disparity_map' in out_dict else None
    if disp is not None:
        disp = depth_to_jet(disp)
        cv2.imwrite(os.path.join(gt_dir_path, f'disparity_{stereo_camera_id}.png'), disp)
    
    seg = out_dict['segmentation_masks'] if 'segmentation_masks' in out_dict else None
    if seg is not None:
        seg = mask_to_color(seg)
        cv2.imwrite(os.path.join(gt_dir_path, f'segmentation_{stereo_camera_id}.png'), seg)

    normal = out_dict['normal_map'] if 'normal_map' in out_dict else None
    if normal is not None:
        normal = ((normal + 1) / 2) * 255
        cv2.imwrite(os.path.join(gt_dir_path, f'normal_{stereo_camera_id}.png'), normal)

    de_masks = out_dict['denoising_mask_map'] if 'denoising_mask_map' in out_dict else None
    if de_masks is not None:
        for i, de_mask in enumerate(de_masks):
            cv2.imwrite(os.path.join(gt_dir_path, f'denoising_mask_{i}_{stereo_camera_id}.png'), de_mask * 255)

    eps = 1e-5
    de_poses = out_dict['denoising_position_map'] if 'denoising_position_map' in out_dict else None
    if de_poses is not None and de_poses != []:
        for i, (de_pos, de_mask) in enumerate(zip(de_poses, de_masks)):
            for j in range(3):
                de_pos[:, :, j] = (de_pos[:, :, j] - de_pos[:, :, j].min()) / (de_pos[:, :, j].max() - de_pos[:, :, j].min() + eps)
            de_pos = np.ascontiguousarray(de_pos[...,:3] * 255, dtype=np.uint8)
            de_pos = de_pos * de_mask[..., None] + 255 * (1 - de_mask[..., None])
            cv2.imwrite(os.path.join(gt_dir_path, f'denoising_position_{i}_{stereo_camera_id}.png'), de_pos)

    de_normals = out_dict['denoising_normal_map'] if 'denoising_normal_map' in out_dict else None
    if de_normals is not None and de_normals != []:
        for i, (de_normal, de_mask) in enumerate(zip(de_normals, de_masks)):
            de_normal = ((de_normal + 1) / 2) * 255
            de_normal = de_normal * de_mask[..., None] + 255 * (1 - de_mask[..., None])
            cv2.imwrite(os.path.join(gt_dir_path, f'denoising_normal_{i}_{stereo_camera_id}.png'), de_normal)

    de_vecs = out_dict['denoising_vector_map'] if 'denoising_vector_map' in out_dict else None
    if de_vecs is not None and de_vecs != []:
        for i, (de_vec, de_mask) in enumerate(zip(de_vecs, de_masks)):
            # de_vec = flow_to_jet(de_vec[:, :, :2])
            backward = de_vec[:, :, :2]
            forward = de_vec[:, :, 2:4]
            backward = flow_to_image(backward)
            forward = flow_to_image(forward)
            backward = backward * de_mask[..., None] + 255 * (1 - de_mask[..., None])
            forward = forward * de_mask[..., None] + 255 * (1 - de_mask[..., None])
            cv2.imwrite(os.path.join(gt_dir_path, f'denoising_vector_backward_{i}_{stereo_camera_id}.png'), backward)
            cv2.imwrite(os.path.join(gt_dir_path, f'denoising_vector_forward_{i}_{stereo_camera_id}.png'), forward)
    
def custom_save_img(depth=None, normal=None, seg=None, disp=None, path=None):
    if depth is not None:
        depth = depth_to_jet(depth)
        cv2.imwrite(f"{path}_depth.png", depth)

    if disp is not None:
        disp = depth_to_jet(disp)
        cv2.imwrite(f"{path}_disparity.png", disp)

    if seg is not None:
        seg = mask_to_color(seg)
        cv2.imwrite(f"{path}_segmentation.png", seg)

    if normal is not None:
        normal = ((normal + 1) / 2) * 255
        cv2.imwrite(f"{path}_normal.png", normal)