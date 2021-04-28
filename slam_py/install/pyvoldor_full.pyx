import numpy as np
cimport numpy as np
from libcpp cimport bool

cdef extern from "../../voldor/py_export.h":
    int py_voldor_wrapper(
        const float* flows, const float* disparity, const float* disparity_pconf,
        const float* depth_priors, const float* depth_prior_poses, const float* depth_prior_pconfs,
        const float fx, const float fy, const float cx, const float cy, const float basefocal,
        const int N, const int N_dp, const int w, const int h,
        const char* config,
        int& n_registered, float* poses, float* poses_covar, float* depth, float* depth_conf)

def voldor(
    np.ndarray[float, ndim=4] flows not None,
    float fx, float fy, float cx, float cy, 
    float basefocal = 0,
    np.ndarray[float, ndim=2] disparity = None,
    np.ndarray[float, ndim=2] disparity_pconf = None,
    np.ndarray[float, ndim=3] depth_priors = None,
    np.ndarray[float, ndim=2] depth_prior_poses = None,
    np.ndarray[float, ndim=3] depth_prior_pconfs = None,
    str config=''):
    
    cdef int N = flows.shape[0]
    cdef int h = flows.shape[1]
    cdef int w = flows.shape[2]
    cdef int N_dp = 0 if depth_priors is None else depth_priors.shape[0]

    flows = np.ascontiguousarray(flows)
    if disparity is not None:
        disparity = np.ascontiguousarray(disparity)
    if depth_priors is not None:
        depth_priors = np.ascontiguousarray(depth_priors)
    if depth_prior_poses is not None:
        depth_prior_poses = np.ascontiguousarray(depth_prior_poses)
    if depth_prior_pconfs is not None:
        depth_prior_pconfs = np.ascontiguousarray(depth_prior_pconfs)
    
    cdef np.ndarray[float, ndim=2, mode='c'] poses = \
        np.ascontiguousarray(np.zeros((N, 6), dtype=np.float32))
    cdef np.ndarray[float, ndim=3, mode='c'] poses_covar = \
        np.ascontiguousarray(np.zeros((N, 6, 6), dtype=np.float32))
    cdef np.ndarray[float, ndim=2, mode='c'] depth = \
        np.ascontiguousarray(np.zeros((h, w), dtype=np.float32))
    cdef np.ndarray[float, ndim=2, mode='c'] depth_conf = \
        np.ascontiguousarray(np.zeros((h, w), dtype=np.float32))

    cdef int n_registered = 0
    py_voldor_wrapper(
                &flows[0,0,0,0],
                &disparity[0,0] if disparity is not None else NULL,
                &disparity_pconf[0,0] if disparity_pconf is not None else NULL,
                &depth_priors[0,0,0] if depth_priors is not None else NULL,
                &depth_prior_poses[0,0] if depth_prior_poses is not None else NULL,
                &depth_prior_pconfs[0,0,0] if depth_prior_pconfs is not None else NULL,
                fx, fy, cx, cy, basefocal,
                N, N_dp, w, h,
                config.encode(),
                n_registered,
                &poses[0,0],
                &poses_covar[0,0,0],
                &depth[0,0],
                &depth_conf[0,0])

    return {'n_registered': n_registered,
            'poses': poses[:n_registered],
            'poses_covar': poses_covar[:n_registered],
            'depth': depth,
            'depth_conf': depth_conf}


cdef extern from "../../frame-alignment/py_export.h":
    int py_falign_wrapper(
        const float* depths, const float* images, const float* weights,
        float* poses_init, float* poses_ret,
        const int* connectivity,
        float* poses_covar, float* scaling_factor,
        float* visibility_mat, float* consistency_mat,
        const int N, const int w, const int h,
        const float fx, const float fy, const float cx, const float cy,
        const float vbf, const float crw,
        const bool optimize_7dof, const bool graduated_optmize,
        const int stride,
        const float consistency_residual_bound,
        const bool debug)

def falign(
    np.ndarray[float, ndim=3] depths not None,
    float fx, float fy, float cx, float cy, 
    np.ndarray[float, ndim=3] weights = None,
    np.ndarray[float, ndim=3] images = None,
    np.ndarray[float, ndim=2] poses_init = None,
    np.ndarray[int, ndim=1] connectivity = None,
    float vbf=1000, float crw=10,
    bool optimize_7dof=False,
    bool graduated_optmize=False,
    int stride=4,
    float consistency_residual_bound=1.0,
    bool debug=False):
    
    cdef int N = depths.shape[0]
    cdef int h = depths.shape[1]
    cdef int w = depths.shape[2]
    

    depths = np.ascontiguousarray(depths)
    if images is not None:
        images = np.ascontiguousarray(images)
    if weights is not None:
        weights = np.ascontiguousarray(weights)
    if poses_init is not None:
        poses_init = np.ascontiguousarray(poses_init)
    if connectivity is not None:
        connectivity = np.ascontiguousarray(connectivity)
    
    
    cdef np.ndarray[float, ndim=2, mode='c'] poses_ret = \
        np.ascontiguousarray(np.zeros((N, 6), dtype=np.float32))
    cdef np.ndarray[float, ndim=3, mode='c'] poses_covar
    if optimize_7dof:
        poses_covar = np.ascontiguousarray(np.zeros((N, 7, 7), dtype=np.float32))
    else:
        poses_covar = np.ascontiguousarray(np.zeros((N, 6, 6), dtype=np.float32))
    cdef np.ndarray[float, ndim=1, mode='c'] scaling_factor = \
        np.ascontiguousarray(np.zeros((N,), dtype=np.float32))
    cdef np.ndarray[float, ndim=2, mode='c'] visibility_mat = \
        np.ascontiguousarray(np.zeros((N, N), dtype=np.float32))
    cdef np.ndarray[float, ndim=2, mode='c'] consistency_mat = \
        np.ascontiguousarray(np.zeros((N, N), dtype=np.float32))
    
    py_falign_wrapper(
                &depths[0,0,0],
                &images[0,0,0] if images is not None else NULL,
                &weights[0,0,0] if weights is not None else NULL,
                &poses_init[0,0] if poses_init is not None else NULL,
                &poses_ret[0,0],
                &connectivity[0] if connectivity is not None else NULL,
                &poses_covar[0,0,0],
                &scaling_factor[0],
                &visibility_mat[0,0],
                &consistency_mat[0,0],
                N, w, h,
                fx, fy, cx, cy,
                vbf, crw,
                optimize_7dof, graduated_optmize,
                stride,
                consistency_residual_bound,
                debug)
    return {'poses_ret': poses_ret, 
            'poses_covar': poses_covar, 
            'scaling_factor': scaling_factor, 
            'visibility_mat': visibility_mat, 
            'consistency_mat': consistency_mat}
    


cdef extern from "../../pose-graph/py_export.h":
    int py_pose_graph_optm_wrapper(
        const int* poses_idx, const float* poses,
        const int* edges_idx, const float* edges_pose, const float* edges_covar,
        float* poses_ret,
        const int n_poses, const int n_edges,
        const bool optimize_7dof,
        const bool debug)

def pgo(
    np.ndarray[float, ndim=2] poses not None, # N*7
    np.ndarray[int, ndim=2] edges_idx not None, # N*2
    np.ndarray[float, ndim=2] edges_pose not None, # N*7
    np.ndarray[float, ndim=3] edges_covar = None, # N*7*7
    np.ndarray[int, ndim=1] poses_idx = None, # N
    bool optimize_7dof = False,
    bool debug = False):
    
    cdef int N_poses = poses.shape[0]
    cdef int N_edges = edges_pose.shape[0]
    
    poses = np.ascontiguousarray(poses)
    edges_idx = np.ascontiguousarray(edges_idx)
    edges_pose = np.ascontiguousarray(edges_pose)
    edges_covar = np.ascontiguousarray(edges_covar)
    
    cdef np.ndarray[float, ndim=2, mode='c'] poses_ret = \
        np.ascontiguousarray(np.zeros((N_poses, 7), dtype=np.float32))
    
    py_pose_graph_optm_wrapper(
                &poses_idx[0] if poses_idx is not None else NULL,
                &poses[0,0],
                &edges_idx[0,0],
                &edges_pose[0,0],
                &edges_covar[0,0,0],
                &poses_ret[0,0],
                N_poses, N_edges,
                optimize_7dof,
                debug)
    
    return poses_ret