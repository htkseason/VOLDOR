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

