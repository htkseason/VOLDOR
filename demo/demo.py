
import argparse
parser = argparse.ArgumentParser(description='VOLDOR-SLAM demo script')
parser.add_argument('--mode', type=str, required=True, help='One from stereo/mono-scaled/mono. For stereo and mono-scaled, disparity input will be required.')
parser.add_argument('--flow_dir', type=str, required=True)
parser.add_argument('--disp_dir', type=str)
parser.add_argument('--img_dir', type=str)
parser.add_argument('--fx', type=float, required=True)
parser.add_argument('--fy', type=float, required=True)
parser.add_argument('--cx', type=float, required=True)
parser.add_argument('--cy', type=float, required=True)
parser.add_argument('--bf', type=float, default=0, help='Baseline x focal, which determines the world scale. If set to 0, default baseline is 0.')
parser.add_argument('--resize', type=float, default=0.5, help='resize input size')
parser.add_argument('--abs_resize', type=float, help='Resize factor related to the size that optical flow is estimated from. (useful to residual model)')
parser.add_argument('--enable_loop_closure', type=str, default=None)
parser.add_argument('--enable_mapping', action='store_true')
parser.add_argument('--save_poses', type=str)
parser.add_argument('--save_depths', type=str)

opt = parser.parse_args()
if opt.abs_resize is None:
    opt.abs_resize = opt.resize

import sys
sys.path.append('../slam_py')
from voldor_viewer import VOLDOR_Viewer
from voldor_slam import VOLDOR_SLAM

import os
import time
import threading

if __name__ == '__main__':
    print(opt)

    # init slam instance and select mode from mono/mono-scaled/stereo
    slam = VOLDOR_SLAM(mode=opt.mode)

    # set camera intrinsic
    slam.set_cam_params(opt.fx,opt.fy,opt.cx,opt.cy,opt.bf, rescale=opt.resize)
    slam.voldor_user_config = f'--abs_resize_factor {opt.abs_resize}'

    # enable loop closure
    if opt.enable_loop_closure is not None:
        slam.enable_loop_closure(opt.enable_loop_closure)

    # start flow loader
    threading.Thread(target=slam.flow_loader, kwargs={'flow_path':opt.flow_dir, 'resize':opt.resize}).start()
    slam.flow_loader_sync(0, block_when_uninit=True)

    # start image loader
    if opt.img_dir is not None:
        threading.Thread(target=slam.image_loader, kwargs={'image_path':opt.img_dir}).start()
        slam.image_loader_sync(0, block_when_uninit=True)
        slam.use_image_info=True
    else:
        slam.use_image_info=False
    
    # start disparity loader
    if opt.disp_dir is not None:
        threading.Thread(target=slam.disp_loader, kwargs={'disp_path':opt.disp_dir}).start()    
        slam.disp_loader_sync(0, block_when_uninit=True)
    
    # start viewer
    viewer = VOLDOR_Viewer(slam)
    viewer_thread = threading.Thread(target=viewer.start)
    viewer_thread.start()
    
    # start VO and mapping threads
    vo_thread = threading.Thread(target=slam.vo_thread)
    vo_thread.start()
    if opt.enable_mapping:
        mapping_thread = threading.Thread(target=slam.mapping_thread)
        mapping_thread.start()

    # wait them to end
    vo_thread.join()
    if opt.enable_mapping:
        mapping_thread.join()

    # save poses and depths
    if opt.save_poses is not None:
        slam.save_poses(opt.save_poses, format='KITTI')
    if opt.save_depths is not None:
        slam.save_depth_maps(opt.save_depths)
