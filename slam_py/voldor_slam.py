import os, sys
import time
import numpy as np
import cv2
from flow_utils import load_flow
try:
    import pyvoldor_full as pyvoldor
    pyvoldor_module = 'full'
    print('Full pyvoldor module loaded.')
except:
    try:
        import pyvoldor_vo as pyvoldor
        pyvoldor_module = 'vo'
        print('VO pyvoldor module loaded.')
    except:
        raise 'Cannot load pyvoldor module.'
from slam_utils import *
import multiprocessing
import threading
from functools import partial
from scipy.spatial.transform import Rotation as Rot
from sklearn.linear_model import HuberRegressor
from rwlock import RWLock

import time
def tic():
    globals()['tictoc'] = time.perf_counter()
def toc():
    return time.perf_counter()-globals()['tictoc']

class Frame:
    def __init__(self, Tcw, depth=None, depth_conf=None, scale=1.0, is_keyframe = False):
        self.Tcw = Tcw.copy()
        self.depth = depth
        self.depth_conf = depth_conf
        self.scale = scale
        self.is_keyframe = is_keyframe

    def get_scaled_depth(self):
        return self.depth * self.scale   


class Edge:
    pose_static = np.zeros((7,), np.float32)
    pose_covar_null = np.diag([1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7]).astype(np.float32) # use when lost tracking
    
    def __init__(self, fid1, fid2, pose, pose_covar, pose_eval_time_scale=1.0, edge_type='vo'):

        
        self.fid1 = fid1
        self.fid2 = fid2

        if pose.shape == (7,):
            self.pose = pose.copy()
        elif pose.shape == (6,):
            self.pose = Edge.pose_static.copy()
            self.pose[:6] = pose
        else:
            raise 'Invalid pose input for Edge'

        if pose_covar.shape == (7,7):
            self.pose_covar = pose_covar.copy()
        elif pose_covar.shape == (6,6):
            self.pose_covar = np.zeros((7,7), np.float32)
            self.pose_covar[:6,:6] = pose_covar
            self.pose_covar[6,6] = ( np.sqrt(pose_covar[3,3]) +
                                    np.sqrt(pose_covar[4,4]) + 
                                    np.sqrt(pose_covar[5,5]) )**2
        else:
            raise 'Invalid pose covar input for Edge'
        
        # ignore dependencies among trans/rot/scale for stability
        self.pose_covar[:3,3:] = 0
        self.pose_covar[3:,:3] = 0
        self.pose_covar[:6,6] = 0
        self.pose_covar[6,:6] = 0

        self.pose[3:6] /= pose_eval_time_scale
        self.pose_covar[3:6,3:6] /= pose_eval_time_scale**2

    
class VOLDOR_SLAM:

    def __init__(self, mode='mono'):
        self.voldor_winsize = 5
        
        # key-frame selection related config
        self.vostep_visibility_thresh = 0.8 # visibility threshold for vo window step
        self.spakf_visibility_thresh = 0.8 # visibility threshold for creating a new spatial keyframe
        self.depth_covis_conf_thresh = 0.1 # depth confidence threshold for estimating covisibility

        # mono-scaled related config
        self.depth_scaling_max_pixels = 10000
        self.depth_scaling_conf_thresh = 0.3

        # voldor related, need modify before set_cam_params()
        self.voldor_pose_sample_min_disp = 1.0
        self.voldor_pose_sample_max_disp = 200.0
        
        # pgo related config
        self.pgo_refine_kf_interval = 10
        self.pgo_local_kf_winsize = 50

        # frame-alignment related config
        self.falign_vbf_factor = 5 # virtual basefocal factor
        self.falign_crw = 10 # color consistency weight
        self.falign_local_link_stride = 4 # sampling stride for local links
        self.falign_local_depth_gblur_width = 3 # depth gblur win size for local links
        self.falign_local_image_gblur_width = 5 # image gblur win size for local links
        self.falign_lc_link_stride = 3 # sampling stride for loop closure links
        self.falign_lc_depth_gblur_width = 5 # depth gblur win size for loop closure links
        self.falign_lc_image_gblur_width = 9 # image gblur win size for loop closure links
        
        # loop closure related config
        self.lc_bow_score_thresh = 0.04
        self.lc_geo_inlier_thresh = 0.4
        self.lc_min_kf_distance = 20
        self.lc_link_visibility_thresh = 0.65
        self.lc_link_consistency_thresh = 0.75

        # mapping related config
        self.mp_realtime_link_thresh = 0.95
        self.mp_no_link_thresh = 0.5
        self.mp_spatial_sigma = 10
        self.mp_temporal_sigma = 30
        self.mp_lc_sigma = 2
        self.mp_link_visibility_thresh = 0.75
        self.mp_link_consistency_thresh = 0.85

        # global config/flags
        self.mode = mode # mode should be either 'mono','mono-scaled' or 'stereo'
        self.use_image_info = True # use color consistency for frame-alignment
        self.end_of_vo = False # flag for indicating vo is end
        self.voldor_user_config = '' # specify parameters for VO
        self.disable_dp = False # disable temporal and spatial depth priors for VO
        self.disable_local_mapping = False

        # internal use
        self._use_loop_closure = False
        self._block_vo_signal = False
        self._map_lock = RWLock()
        self._viewer_signal_map_changed = False

        if mode=='stereo':
            self.voldor_config = '--silent --meanshift_kernel_var 0.1 --disp_delta 1 --delta 0.2 --max_iters 4 '
            self.mp_realtime_link_thresh = 1
            self.pgo_refine_kf_interval = 20
        elif mode=='mono-scaled':
            self.voldor_config = '--silent --meanshift_kernel_var 0.2 --delta 1.5 --max_iters 5 '
            self.mp_realtime_link_thresh = 1
            self.pgo_refine_kf_interval = 20
        elif mode=='mono':
            self.voldor_config = '--silent --meanshift_kernel_var 0.2 --delta 1.5 --max_iters 5 '
            self.mp_realtime_link_thresh = 0.95
            self.pgo_refine_kf_interval = 10
        else:
            raise f'Unknown SLAM mode - {mode}'

        self.flows = []
        self.images_grayf = [] # gray-float image for frame-alignment
        self.images_bgri = [] # bgr-uint8 image for viewer rendering
        self.disps = []
        self.flow_loader_pt = -1
        self.image_loader_pt = -1
        self.disp_loader_pt = -1
        self.lc_candidates = []

        self.fx,fy,cx,cy = 0,0,0,0
        self.basefocal = 0
        self.N_FRAMES = float('nan')
        
        self.fid_cur = 0
        self.fid_cur_tmpkf = -1 # temporal key frame
        self.fid_cur_spakf = -1 # spatial key frame
        self.Twc_cur = np.eye(4,4,dtype=np.float32)

        self.frames = []
        self.edges = []
        self.kf_ids = []
        
        if os.name=='nt':
            # For multi-threading performance,
            # We let viewer live in main process to share all large data memory of slam instance
            # Other slow function calls (pyvoldor, pyfalign, pypgo) will run at remote processes
            self.cython_process_pool = multiprocessing.Pool(6)
            self.falign_thread_pool = multiprocessing.pool.ThreadPool(12)
        else:
            # Linux remote processing have problem with some opencv builds...
            self.cython_process_pool = multiprocessing.pool.ThreadPool(6)
            self.falign_thread_pool = multiprocessing.pool.ThreadPool(12)

    def set_cam_params(self, fx, fy, cx, cy, basefocal='auto', rescale=1.0):
        self.fx = fx*rescale
        self.fy = fy*rescale
        self.cx = cx*rescale
        self.cy = cy*rescale
        if basefocal == 'auto' or basefocal<=0:
            self.basefocal = (fx+fy)*0.25*rescale # default virtual bf = 0.5*focal
        else:
            self.basefocal = basefocal*rescale
        self.K = np.array([[fx,0,cx], [0,fy,cy], [0,0,1]], np.float32)
        self.K_inv = np.linalg.inv(self.K)
        self.voldor_config += f'--pose_sample_min_depth {self.basefocal/self.voldor_pose_sample_max_disp} --pose_sample_max_depth {self.basefocal/self.voldor_pose_sample_min_disp} '
        print(f'Camera parameters set to {self.fx}, {self.fy}, {self.cx}, {self.cy}, {self.basefocal}')

    def flow_loader_sync(self, fid_query, no_block=False, block_when_uninit=False):
        if (self.flow_loader_pt == -1 and not block_when_uninit) \
            or fid_query >= self.N_FRAMES-1:
            return False
        while self.flow_loader_pt <= fid_query:
            if no_block:
                return False
            time.sleep(0.01)
        return True
    def image_loader_sync(self, fid_query, no_block=False, block_when_uninit=False):
        if (self.image_loader_pt == -1 and not block_when_uninit) \
            or fid_query >= self.N_FRAMES-1:
            return False
        while self.image_loader_pt <= fid_query:
            if no_block:
                return False
            time.sleep(0.01)
        return True
    def disp_loader_sync(self, fid_query, no_block=False, block_when_uninit=False):
        if (self.disp_loader_pt == -1 and not block_when_uninit) \
            or fid_query >= self.N_FRAMES-1:
            return False
        while self.disp_loader_pt <= fid_query:
            if no_block:
                return False
            time.sleep(0.01)
        return True
        
    def flow_loader(self, flow_path, resize=1.0, n_pre_cache=100, range=(0,0)):
        self.flow_loader_pt = 0

        flow_fn_list = sorted(os.listdir(flow_path))
        if range != (0,0):
            flow_fn_list = flow_fn_list[range[0]:range[1]]
        print(f'{len(flow_fn_list)} flows loaded')
        flow_example = load_flow(os.path.join(flow_path, flow_fn_list[0]))
        self.N_FRAMES = len(flow_fn_list)+1
        self.h = int(flow_example.shape[0]*resize)
        self.w = int(flow_example.shape[1]*resize)
        
        for fn in flow_fn_list:
            while len(self.flows) - self.fid_cur > n_pre_cache:
                time.sleep(0.01)

            flow = load_flow(os.path.join(flow_path, fn))
            if flow.shape[0] != self.h or flow.shape[1] != self.w:
                flow_rescale = (self.w / flow.shape[1], self.h / flow.shape[0])
                flow = cv2.resize(flow, (self.w, self.h))
                flow[...,0] *= flow_rescale[0]
                flow[...,1] *= flow_rescale[1]
            self.flows.append(flow)
            self.flow_loader_pt += 1
            
    def image_loader(self, image_path, n_pre_cache=100, range=(0,0)):
        self.image_loader_pt = 0

        image_fn_list = sorted(os.listdir(image_path))
        if range!=(0,0):
            image_fn_list = image_fn_list[range[0]:range[1]]
        print(f'{len(image_fn_list)} images loaded')
        
        for fn in image_fn_list:
            while len(self.images_grayf) - self.fid_cur > n_pre_cache or self.flow_loader_pt <= 0:
                time.sleep(0.01)

            img = cv2.imread(os.path.join(image_path, fn), cv2.IMREAD_COLOR)

            if img.shape[0] != self.h or img.shape[1] != self.w:
                img = cv2.resize(img, (self.w, self.h))

            self.images_bgri.append(img.copy())

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype(np.float32) / 255.0
            self.images_grayf.append(img)
            self.image_loader_pt += 1

    def disp_loader(self, disp_path, n_pre_cache=100, range=(0,0)):
        self.disp_loader_pt = 0

        disp_fn_list = sorted(os.listdir(disp_path))
        if range!=(0,0):
            disp_fn_list = disp_fn_list[range[0]:range[1]]
        print(f'{len(disp_fn_list)} disparities loaded')
        
        for fn in disp_fn_list:
            while len(self.disps) - self.fid_cur > n_pre_cache or self.flow_loader_pt <= 0:
                time.sleep(0.01)

            if fn.endswith('.flo'):
                disp = -load_flow(os.path.join(disp_path, fn))[...,0]
                disp = np.ascontiguousarray(disp)
            elif fn.endswith('.png'):
                disp = cv2.imread(os.path.join(disp_path, fn), cv2.IMREAD_UNCHANGED)
                disp = disp.astype(np.float32) / 256.0
            else:
                raise f'Unsupported disparity format {fn}'

            if disp.shape[0] != self.h or disp.shape[1] != self.w:
                disp_rescale = self.w / disp.shape[1]
                disp = cv2.resize(disp, (self.w, self.h)) * disp_rescale
            self.disps.append(disp)
            self.disp_loader_pt += 1

    def save_poses(self, file_path='./output_pose.txt', format='KITTI'):
        with open(file_path, 'w') as f:
            for fid in range(self.N_FRAMES):
                if format == 'KITTI':
                    Tcw34 = self.frames[fid].Tcw[:3,:4].reshape(-1)
                    f.write(' '.join([str(v) for v in Tcw34]))
                    f.write('\n')
                elif format == 'TartanAir':
                    r = Rot.from_matrix(self.frames[fid].Tcw[:3,:3])
                    r_quat = r.as_quat()
                    t = self.frames[fid].Tcw[:3,3]
                    f.write(f'{t[2]} {t[0]} {t[1]} {r_quat[2]} {r_quat[0]} {r_quat[1]} {r_quat[3]}\n')
        print(f'Camera poses saved to {file_path} with {format} format')

    def save_depth_maps(self, save_dir='./depths', zfill=6):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        for fid in self.kf_ids:
            np.save(os.path.join(save_dir, f'{str(fid).zfill(zfill)}_depth.npy'), self.frames[fid].get_scaled_depth())
            np.save(os.path.join(save_dir, f'{str(fid).zfill(zfill)}_depth_conf.npy'), self.frames[fid].depth_conf)
        print(f'{len(self.kf_ids)} depth maps saved to {save_dir}')


    def enable_loop_closure(self, voc_path='./ORBvoc.bin'):
        if pyvoldor_module != 'full':
            print('Error: Loop closure not available. Need full pyvoldor module.')
            return
        try:
            import pyDBoW3 as bow
        except:
            print('Error: Loop closure not available. Cannot load pyDBoW3 module.')
            return
        
        voc = bow.Vocabulary()
        print(f'Loading vocabulary from {voc_path} ...')
        voc.load(voc_path) 
        self.bow_db = bow.Database()
        self.bow_db.setVocabulary(voc, True, 0)
        
        self.feature_detector = cv2.ORB_create()
        #self.feature_detector = cv2.AKAZE_create()
        del voc
        self._use_loop_closure = True

    def solve_pgo(self, fid_start=0):
        # note that py_pgo set first frame pose as constant
        # thus optimization must done from start_idx to the end
        with self._map_lock.w_locked():
            assert len(self.frames) == self.fid_cur # make sure Tcw_cur is right after all registered frames
            n_frames_total = len(self.frames) + 1 # +1 for Tcw_cur
            n_edges_total = len(self.edges)
            n_frames = n_frames_total - fid_start
            if n_frames <= 0:
                return
            n_edges = 0

            poses_idx = np.zeros((n_frames,), np.int32)
            poses = np.zeros((n_frames,7), np.float32)
            edges_idx = np.zeros((n_edges_total,2), np.int32)
            edges_pose = np.zeros((n_edges_total,7), np.float32)
            edges_covar = np.zeros((n_edges_total,7,7), np.float32)

            for i in range(fid_start, n_frames_total-1):
                poses_idx[i-fid_start] = i
                poses[i-fid_start,:6] = T44_to_T6(self.frames[i].Tcw)
                poses[i-fid_start,6] = np.log(self.frames[i].scale)
            poses_idx[n_frames-1] = n_frames_total-1
            poses[n_frames-1,:6] = T44_to_T6(np.linalg.inv(self.Twc_cur))
            poses[n_frames-1,6] = np.log(self.frames[n_frames_total-2].scale)
            
            for i in range(n_edges_total):
                if fid_start <= self.edges[i].fid1 < n_frames_total and \
                    fid_start <= self.edges[i].fid2 < n_frames_total:
                    edges_idx[n_edges,0] = self.edges[i].fid1
                    edges_idx[n_edges,1] = self.edges[i].fid2
                    edges_pose[n_edges] = self.edges[i].pose
                    edges_covar[n_edges] = self.edges[i].pose_covar
                    n_edges += 1
            if n_edges == 0:
                return
            
            py_pgo_kwargs = {
                'poses': poses,
                'poses_idx': poses_idx,
                'edges_idx': edges_idx[:n_edges],
                'edges_pose': edges_pose[:n_edges],
                'edges_covar': edges_covar[:n_edges],
                'optimize_7dof': True if self.mode =='mono' else False,
                'debug': False}
                
            py_pgo_funmap = partial(pyvoldor.pgo, **py_pgo_kwargs)
            poses_ret = self.cython_process_pool.apply(py_pgo_funmap)

            for i in range(n_frames-1):
                self.frames[i+fid_start].Tcw = T6_to_T44(poses_ret[i,:6])
                self.frames[i+fid_start].scale = np.exp(poses_ret[i,6])
            self.Twc_cur = np.linalg.inv(T6_to_T44(poses_ret[n_frames-1,:6]))
            print(f'solve pgo {fid_start}-{n_frames_total}, n_frames={n_frames}, n_edges={n_edges}')

    def process_vo(self):
        # a read lock will work since the 'write' of VO is 'append' that does not change existing map
        with self._map_lock.r_locked():
            if self.fid_cur >= (self.N_FRAMES - 1): # since n_flow = n_frames-1
                self.frames.append(Frame(np.linalg.inv(self.Twc_cur))) # last frame
                self.fid_cur = self.N_FRAMES
                return False
            
            # prepare depth priors from temporal and spatial kf
            depth_priors = []
            depth_prior_pconfs = []
            depth_prior_poses = []
            dpkf_list = []
            
            if not self.disable_dp:
                if self.fid_cur_tmpkf >= 0:
                    dpkf_list.append(self.fid_cur_tmpkf)
                if self.fid_cur_spakf >= 0 and self.fid_cur_spakf != self.fid_cur_tmpkf:
                    dpkf_list.append(self.fid_cur_spakf)

            for fid in dpkf_list:
                if fid >= 0:
                    depth_priors.append(self.frames[fid].get_scaled_depth())
                    depth_prior_pconfs.append(self.frames[fid].depth_conf)
                    depth_prior_poses.append(T44_to_T6(np.linalg.inv(self.Twc_cur @ self.frames[fid].Tcw)))
            
            if not self.flow_loader_sync(min(self.fid_cur+self.voldor_winsize-1, self.N_FRAMES-2)):
                raise 'Flow loader not working or files are missing.'
            if self.mode=='stereo':
                if not self.disp_loader_sync(self.fid_cur):
                    raise 'Disparity loader not working or files are missing.'
            py_voldor_kwargs = {
                'flows': np.stack(self.flows[self.fid_cur:self.fid_cur+self.voldor_winsize], axis=0),
                'fx':self.fx, 'fy':self.fy, 'cx':self.cx, 'cy':self.cy, 'basefocal':self.basefocal,
                'disparity' : self.disps[self.fid_cur] if self.mode=='stereo' else None,
                'depth_priors' : np.stack(depth_priors, axis=0) if len(depth_priors)>0 else None,
                'depth_prior_pconfs' : np.stack(depth_prior_pconfs, axis=0) if len(depth_prior_pconfs)>0 else None,
                'depth_prior_poses' : np.stack(depth_prior_poses, axis=0) if len(depth_prior_poses)>0 else None,
                'config' : self.voldor_config + ' ' + self.voldor_user_config}

            py_voldor_funmap = partial(pyvoldor.voldor, **py_voldor_kwargs)
            vo_ret = self.cython_process_pool.apply(py_voldor_funmap)

            
            # if vo failed
            if vo_ret['n_registered'] == 0:
                print(f'Tracking lost at {self.fid_cur}')

                self.frames.append(Frame(np.linalg.inv(self.Twc_cur)))
                self.edges.append(Edge(self.fid_cur, self.fid_cur+1, pose=Edge.pose_static, pose_covar=Edge.pose_covar_null, edge_type='none'))

                self.fid_cur_tmpkf = -1
                self.fid_cur_spakf = -1
                self.fid_cur = self.fid_cur + 1
                
            else:
                if self.mode=='mono-scaled':
                    if not self.disp_loader_sync(self.fid_cur):
                        raise 'Disparity loader not working or files are missing.'
                    mask = vo_ret['depth_conf']>self.depth_scaling_conf_thresh
                    src = self.basefocal / vo_ret['depth'][mask]
                    dst = self.disps[self.fid_cur][mask]
                    
                    if src.size > self.depth_scaling_max_pixels:
                        indices = np.arange(src.size)
                        np.random.shuffle(indices)
                        src=src[indices[:self.depth_scaling_max_pixels]]
                        dst=dst[indices[:self.depth_scaling_max_pixels]]
                    
                    huber = HuberRegressor(fit_intercept=False)
                    huber = huber.fit(src.reshape(-1,1),dst)
                    scale = max(min(1.0 / huber.coef_, 10), 0.1)                    
                    
                    vo_ret['depth'] *= scale
                    vo_ret['poses'][:,3:6] *= scale
                    vo_ret['poses_covar'][:,:,3:6] *= scale
                    vo_ret['poses_covar'][:,3:6,:] *= scale

                vo_ret['Tc1c2'] = T6_to_T44(vo_ret['poses'])

                # based on covisibility, figure out how many steps to move
                vo_step = 0
                T_tmp = np.eye(4,4,dtype=np.float32)
                for i in range(vo_ret['n_registered']):
                    vo_step = vo_step + 1
                    T_tmp = vo_ret['Tc1c2'][i] @ T_tmp
                    covis = eval_covisibility(vo_ret['depth'], T_tmp, self.K, vo_ret['depth_conf']>self.depth_covis_conf_thresh)
                    if covis < self.vostep_visibility_thresh:
                        break
            
                # insert frames and edges, move Twc_cur
                for i in range(vo_step):
                    if i==0:
                        self.frames.append(Frame(np.linalg.inv(self.Twc_cur), vo_ret['depth'], vo_ret['depth_conf']))
                    else:
                        self.frames.append(Frame(np.linalg.inv(self.Twc_cur)))

                    self.edges.append(Edge(self.fid_cur+i, self.fid_cur+i+1, pose=vo_ret['poses'][i], \
                            pose_covar=vo_ret['poses_covar'][i], \
                            pose_eval_time_scale=self.frames[self.fid_cur_tmpkf].scale,
                            edge_type='vo'))

                    self.Twc_cur = vo_ret['Tc1c2'][i] @ self.Twc_cur
                    polish_T44(self.Twc_cur)
                
                # based on covisibility, to see if need let current frame be a new spatial keyframe
                if self.fid_cur_spakf >= 0:
                    T_spa2cur = self.Twc_cur @ self.frames[self.fid_cur_spakf].Tcw
                    covis = eval_covisibility(self.frames[self.fid_cur_spakf].get_scaled_depth(), T_spa2cur, self.K, self.frames[self.fid_cur_spakf].depth_conf > self.depth_covis_conf_thresh)
                    if covis < self.spakf_visibility_thresh:
                        self.append_kf(self.fid_cur)
                        self.fid_cur_spakf = self.fid_cur
                else:
                    self.append_kf(self.fid_cur)
                    self.fid_cur_spakf = self.fid_cur

                # set temporal kf to current frame, move fid_cur pt
                self.fid_cur_tmpkf = self.fid_cur
                self.fid_cur = self.fid_cur + vo_step

        return True

    def establish_local_links(self, kf_ids):
        # a read lock will work since the process does not change the map
        with self._map_lock.r_locked():
            depths = []
            weights = []
            poses_init = []
            images = []

            for fid in kf_ids:
                depth = self.frames[fid].get_scaled_depth()
                depth = cv2.GaussianBlur(depth, (self.falign_local_depth_gblur_width, self.falign_local_depth_gblur_width),0)
                depths.append(depth)
                weights.append(self.frames[fid].depth_conf)
                #poses_init.append(T44_to_T6(self.frames[fid].Tcw))
                poses_init.append(T44_to_T6(np.linalg.inv(self.frames[kf_ids[0]].Tcw) @ self.frames[fid].Tcw))
                    
                if self.use_image_info:
                    if not self.image_loader_sync(fid):
                        raise 'Image loader not working or files are missing.'
                    image = cv2.GaussianBlur(self.images_grayf[fid],(self.falign_local_image_gblur_width, self.falign_local_image_gblur_width),0)
                    images.append(image)

            py_falign_kwargs = {
                'depths': np.stack(depths, axis=0),
                'fx':self.fx, 'fy':self.fy, 'cx':self.cx, 'cy':self.cy, 
                'weights': np.stack(weights, axis=0),
                'poses_init': np.stack(poses_init, axis=0),
                'images': np.stack(images, axis=0) if self.use_image_info else None,
                'optimize_7dof': True if self.mode=='mono' else False,
                'stride': self.falign_local_link_stride,
                'vbf': self.basefocal*self.falign_vbf_factor,
                'crw': self.falign_crw,
                'debug': False}

            py_falign_funmap = partial(pyvoldor.falign, **py_falign_kwargs)
            falign_ret = self.cython_process_pool.apply(py_falign_funmap)
            #print(falign_ret['scaling_factor'])
            consistency = np.mean(falign_ret['consistency_mat'][np.isfinite(falign_ret['consistency_mat'])])
            visibility = np.mean(falign_ret['visibility_mat'][np.isfinite(falign_ret['visibility_mat'])])
            #print(consistency, visibility)
            if consistency < self.mp_link_consistency_thresh or visibility < self.mp_link_visibility_thresh:
                return
            if np.any(np.linalg.matrix_rank(falign_ret['poses_covar']) != falign_ret['poses_covar'].shape[1]):
                return
    
            falign_ret['Tcw'] = T6_to_T44(falign_ret['poses_ret'])
            # links are fully-connected
            for i1 in range(len(kf_ids)-1):
                for i2 in range(i1+1,len(kf_ids)):
                    Tc1c2 = np.linalg.inv(falign_ret['Tcw'][i2]) @ falign_ret['Tcw'][i1]
                    pose7 = np.zeros((7,), np.float32)
                    pose7[:6] = T44_to_T6(Tc1c2)
                    f1_scale = self.frames[kf_ids[i1]].scale*falign_ret['scaling_factor'][i1]
                    f2_scale = self.frames[kf_ids[i2]].scale*falign_ret['scaling_factor'][i2]                
                    pose7[6] = np.log(f2_scale / f1_scale)
                    pose_eval_time_scale = np.sqrt(f1_scale*f2_scale)
                    self.edges.append(Edge(kf_ids[i1],kf_ids[i2], pose7, falign_ret['poses_covar'][i2], pose_eval_time_scale=pose_eval_time_scale, edge_type='falign-local'))


    def establish_lc_links(self, kf_ids):
        print('Loop closure at ', kf_ids)
        # a read lock will work since the process does not change the map
        with self._map_lock.r_locked():
            depths = []
            depths_median_scaling = []
            weights = []
            #poses_init = []
            images = []

            for fid in kf_ids:
                depth = self.frames[fid].get_scaled_depth()
                if self.mode=='mono':
                    scaling = 10/np.median(depth)
                    depths_median_scaling.append(scaling)
                    depth *= scaling
                depth = cv2.GaussianBlur(depth, (self.falign_lc_depth_gblur_width,self.falign_lc_depth_gblur_width),0)
                depths.append(depth)
                weights.append(self.frames[fid].depth_conf)
                #poses_init.append(np.zeros((6,),np.float32))

                if self.use_image_info:
                    if not self.image_loader_sync(fid):
                        raise 'Image loader not working or files are missing.'
                    image = cv2.GaussianBlur(self.images_grayf[fid],(self.falign_lc_image_gblur_width,self.falign_lc_image_gblur_width),0)
                    images.append(image)

            py_falign_kwargs = {
                'depths': np.stack(depths, axis=0),
                #'images': np.stack(images, axis=0) if self.use_image_info else None,
                'fx':self.fx, 'fy':self.fy, 'cx':self.cx, 'cy':self.cy, 
                'weights': np.stack(weights, axis=0),
                #'poses_init': np.stack(poses_init, axis=0),
                'optimize_7dof': True if self.mode=='mono' else False,
                'stride': self.falign_lc_link_stride,
                'vbf': self.basefocal*self.falign_vbf_factor,
                'crw': self.falign_crw,
                'debug': False}
                #'debug': True}

            py_falign_funmap = partial(pyvoldor.falign, **py_falign_kwargs)
            falign_ret = self.cython_process_pool.apply(py_falign_funmap)
            
            # refine alignment with image
            if self.use_image_info:
                py_falign_kwargs['images'] = np.stack(images, axis=0)
                py_falign_kwargs['poses_init'] = falign_ret['poses_ret']
                py_falign_funmap = partial(pyvoldor.falign, **py_falign_kwargs)
                falign_ret = self.cython_process_pool.apply(py_falign_funmap)

            consistency = np.mean(falign_ret['consistency_mat'][np.isfinite(falign_ret['consistency_mat'])])
            visibility = np.mean(falign_ret['visibility_mat'][np.isfinite(falign_ret['visibility_mat'])])

            #print(consistency, visibility)
            if consistency < self.lc_link_consistency_thresh or visibility < self.lc_link_visibility_thresh:
                print(f'Loop closure registration score = {consistency:.4f} / {visibility:.4f}, rejected')
                return

            if np.any(np.linalg.matrix_rank(falign_ret['poses_covar']) != falign_ret['poses_covar'].shape[1]):
                return
            
            print(f'Loop closure registration score = {consistency:.4f} / {visibility:.4f}')

            falign_ret['Tcw'] = T6_to_T44(falign_ret['poses_ret'])
            if self.mode == 'mono':
                for i in range(len(falign_ret['scaling_factor'])):
                    falign_ret['scaling_factor'][i] *= depths_median_scaling[i]
            
            # links are fully-connected
            for i1 in range(len(kf_ids)-1):
                for i2 in range(i1+1,len(kf_ids)):
                    Tc1c2 = np.linalg.inv(falign_ret['Tcw'][i2]) @ falign_ret['Tcw'][i1]
                    pose7 = np.zeros((7,), np.float32)
                    pose7[:6] = T44_to_T6(Tc1c2)
                    f1_scale = self.frames[kf_ids[i1]].scale*falign_ret['scaling_factor'][i1]
                    f2_scale = self.frames[kf_ids[i2]].scale*falign_ret['scaling_factor'][i2]                
                    pose7[6] = np.log(f2_scale / f1_scale)
                    pose_eval_time_scale = np.sqrt(f1_scale*f2_scale)
                    self.edges.append(Edge(kf_ids[i1],kf_ids[i2], pose7, falign_ret['poses_covar'][i2], pose_eval_time_scale=pose_eval_time_scale, edge_type='falign-lc'))


    def append_kf(self, fid):

        self.frames[fid].is_keyframe = True
        self.kf_ids.append(fid)

        if self._use_loop_closure:
            if not self.image_loader_sync(fid):
                raise 'Image loader not working or files are missing.'
            #tic()
            kps, des = self.feature_detector.detectAndCompute(self.images_bgri[fid], None)
            self.frames[fid].kps = kps
            self.frames[fid].des = des
            ret = self.bow_db.query(des,-1,-1)
            for r in ret:
                if r.Score > self.lc_bow_score_thresh:
                    if len(self.kf_ids) - r.Id < self.lc_min_kf_distance:
                        continue
                    inlier_rate = geometry_check(self.frames[fid].kps, self.frames[fid].des, \
                                              self.frames[self.kf_ids[r.Id]].kps, self.frames[self.kf_ids[r.Id]].des)
                    if inlier_rate > self.lc_geo_inlier_thresh:
                        self.lc_candidates.append((r.Id, len(self.kf_ids)-1)) # register kf-id
                        #print('lc candidate ', fid, self.kf_ids[r.Id], r.Score, inlier_rate)
                    
            self.bow_db.add(des)
            #img_kp=cv2.drawKeypoints(self.images_bgri[fid],kps,None,color=(0,255,0))            
            #cv2.imshow('imgkp',img_kp)
            #cv2.waitKey(1)
            #toc()


    def vo_thread(self):
        print('VO thread started')
        print(f'VO mode = {self.mode}')
        self.end_of_vo = False
        while self.process_vo():
            self._viewer_signal_map_changed = True
            #if self.fid_cur_tmpkf>=0:
                #print('current scale = ', self.frames[0].scale,self.frames[-1].scale)
            #print(f'{self.fid_cur}  <-  {self.fid_cur_tmpkf}, {self.fid_cur_spakf}')
            if self.fid_cur_tmpkf >= 0:
                cv2.imshow('tmpkf_depth', (self.basefocal*0.04)/self.frames[self.fid_cur_tmpkf].get_scaled_depth())
                cv2.imshow('tmpkf_depth_conf', self.frames[self.fid_cur_tmpkf].depth_conf)
                if cv2.waitKey(1) == 113:
                    os._exit(1)
            while self._block_vo_signal:
                time.sleep(0.01)
        self.end_of_vo = True
        print('VO thread ended.')
        print(f'{len(self.kf_ids)} keyframes registered.')


    def mapping_thread(self):
        if pyvoldor_module != 'full':
            print('Error: Mapping not available. Need full pyvoldor module.')
            return
        print('Mapping thread started')
        n_kfs_registered = 0
        next_pgo_kfid = self.pgo_refine_kf_interval
        link_mask = np.zeros((self.N_FRAMES,self.N_FRAMES), np.bool) # already matched mask
        priority_mat = np.zeros((self.N_FRAMES, self.N_FRAMES), np.float32)
        lc_pairs = set()
        new_local_link_flag = False
        new_lc_link_flag = False
        while not self.end_of_vo or n_kfs_registered<len(self.kf_ids):
            n_kfs_cur = len(self.kf_ids)

            if n_kfs_cur == 0:
                time.sleep(0.01)
                continue
            
            if n_kfs_registered == n_kfs_cur:
                # if all kfs already registered, link the edge with highest priority
                Iy, Ix = np.unravel_index(np.argmax(priority_mat), priority_mat.shape)
                if priority_mat[Iy,Ix] > self.mp_no_link_thresh and not link_mask[Iy,Ix]:
                    if (Iy,Ix) in lc_pairs:
                        new_lc_link_flag = True
                        self.establish_lc_links([self.kf_ids[Iy],self.kf_ids[Ix]])
                    else:
                        new_local_link_flag = True
                        self.establish_local_links([self.kf_ids[Iy],self.kf_ids[Ix]])
                    link_mask[Iy,Ix] = True
                    priority_mat[Iy,Ix] = 0
                time.sleep(0.01)
            else:
                # if new kf exists, update priority map, and link realtime required edges
                # set block_vo signal to soft-block vo thread and wait for realtime required edges
                self._block_vo_signal = True

                priority_mat[...] = 0

                if not self.disable_local_mapping:
                    for f1 in range(max(0,n_kfs_cur-2*self.mp_temporal_sigma), n_kfs_cur):
                        for f2 in range(f1+1,min(n_kfs_cur,f1+2*self.mp_spatial_sigma)):
                            priority_mat[f1, f2] = max(priority_mat[f1, f2], np.exp( - ((f1-f2)/self.mp_spatial_sigma)**2  - ((n_kfs_cur-f1)*(n_kfs_cur-f2) / self.mp_temporal_sigma**2)))

                # since register lc links are expensive (w/o initialization)
                # we only establish links within a small neighborhoods
                for f1,f2 in self.lc_candidates:
                    for ff1, ff2 in [(f1,f2), (f1+1,f2), (f1-1,f2), (f1,f2+1), (f1,f2-1)]:
                        if ff1>=0 and ff1<n_kfs_cur and ff2>=0 and ff2<n_kfs_cur:
                            priority_mat[ff1, ff2] = max(priority_mat[ff1, ff2], np.exp(- ( (abs(ff1-f1)+abs(ff2-f2)) / self.mp_lc_sigma)**2))
                            lc_pairs.add((ff1,ff2))

                priority_mat[link_mask] = 0

                Iy,Ix = np.where(priority_mat>self.mp_realtime_link_thresh)

                # establish realtime required links
                if Iy.size>0 and Ix.size>0:
                    falign_task_pool = []
                    for y,x in zip(Iy,Ix):
                        if (y,x) in lc_pairs:
                            new_lc_link_flag = True
                            task = self.falign_thread_pool.apply_async(self.establish_lc_links, ([self.kf_ids[y],self.kf_ids[x]],))
                            falign_task_pool.append(task)
                        else:
                            new_local_link_flag = True
                            task = self.falign_thread_pool.apply_async(self.establish_local_links, ([self.kf_ids[y],self.kf_ids[x]],))
                            falign_task_pool.append(task)
                        link_mask[y,x] = True
                        priority_mat[y,x] = 0
                    for task in falign_task_pool:
                        task.get()

                # solve pose graph
                if (n_kfs_cur>=next_pgo_kfid) and (new_local_link_flag or new_lc_link_flag):
                    if new_lc_link_flag:
                        self.solve_pgo()
                    else:
                        self.solve_pgo(self.kf_ids[0 if self.pgo_local_kf_winsize > n_kfs_cur else -self.pgo_local_kf_winsize])
                    self._viewer_signal_map_changed = True
                    new_local_link_flag = False
                    new_lc_link_flag = False
                    next_pgo_kfid = n_kfs_cur + self.pgo_refine_kf_interval

                n_kfs_registered = n_kfs_cur

                self._block_vo_signal = False


            #priority_mat_show = cv2.resize(priority_mat[:n_kfs_cur,:n_kfs_cur],(800,800),interpolation=cv2.INTER_NEAREST)
            #link_mask_show = cv2.resize(link_mask[:n_kfs_cur,:n_kfs_cur].astype(np.float32),(800,800),interpolation=cv2.INTER_NEAREST)
            #cv2.imshow('priority_mat_show',priority_mat_show)
            #cv2.imshow('link_mask',link_mask_show)
            #if cv2.waitKey(1)=='q':
                #os._exit(1)

        # global pgo after all finished
        self.solve_pgo()
        self._viewer_signal_map_changed = True
        print('Mapping thread end.')



