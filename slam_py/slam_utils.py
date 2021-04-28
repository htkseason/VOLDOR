import numpy as np
import cv2

def geometry_check(kp1, des1, kp2, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.knnMatch(des1,des2, k=1)
    pts1 = []
    pts2 = []
    for m in matches:
        if len(m) > 0:
            pts1.append(kp1[m[0].queryIdx].pt)
            pts2.append(kp2[m[0].trainIdx].pt)
    pts1 = np.array(pts1, np.float32)
    pts2 = np.array(pts2, np.float32)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
    return (2*np.sum(mask)) / (len(kp1) + len(kp2))

def eval_covisibility(depth, Tc1c2, K, mask=None, stride=4):
    if not hasattr(eval_covisibility, 'cache_shape') or \
        eval_covisibility.cache_shape != depth.shape or \
        eval_covisibility.cache_stride != stride or \
        not np.array_equal(eval_covisibility.cache_K, K):

        eval_covisibility.cache_shape = depth.shape
        eval_covisibility.cache_stride = stride
        eval_covisibility.cache_K = K
        #eval_covisibility.cache_K_inv = np.linalg.inv(K)
        h, w = depth.shape
        Iy, Ix = np.mgrid[0:h:stride, 0:w:stride]
        coords_2d = np.stack([Ix, Iy, np.ones_like(Ix)], axis=2).astype(np.float32)
        coords_2d = coords_2d.reshape(-1, 3)
        eval_covisibility.coords_3d_shared = (np.linalg.inv(K) @ coords_2d.T).T

    h, w = depth.shape

    coords_3d = np.copy(eval_covisibility.coords_3d_shared) * depth[::stride,::stride].reshape(-1,1)
    if mask is not None:
        coords_3d = coords_3d[mask[::stride,::stride].reshape(-1)]
    coords_3d = (Tc1c2[:3,:3] @ coords_3d.T).T
    coords_3d = coords_3d + Tc1c2[:3,3]
    
    coords_2d_proj = (eval_covisibility.cache_K @ coords_3d.T).T
    coords_2d_proj = coords_2d_proj[coords_2d_proj[:,2]>0]
    coords_2d_proj = coords_2d_proj[:,:2] / coords_2d_proj[:,2:3]

    # how many pixels are visibilble in the target frame
    visibility = (coords_2d_proj[:,0]>0) & (coords_2d_proj[:,0]<w) & (coords_2d_proj[:,1]>0) & (coords_2d_proj[:,1]<h)
    visibility = np.sum(visibility) / ((w//stride)*(h//stride))
    # how large the area covered by the pixels
    coverage,_,_ = np.histogram2d(coords_2d_proj[:,0],coords_2d_proj[:,1],bins=(w//(2*stride),h//(2*stride)),range=((0,w),(0,h)))
    coverage = np.sum(coverage>0) / ((w//(2*stride))*(h//(2*stride)))
    covisibility_score = 2*(visibility*coverage)/ max(visibility+coverage, 1)
    return covisibility_score

def polish_T44(pose):
    u,s,vt = np.linalg.svd(pose[:3,:3])
    pose[:3,:3] = u @ vt

def T44_to_T6(poses):
    if len(poses.shape) == 2:
        ret = np.zeros((6,),poses.dtype)
        rvec,_ = cv2.Rodrigues(poses[:3,:3])
        ret[:3] = rvec.reshape(-1)
        ret[3:] = poses[:3,3]
        return ret
    elif len(poses.shape) == 3:
        N = poses.shape[0]
        ret = np.zeros((N,6),poses.dtype)
        for i in range(N):
            rvec,_ = cv2.Rodrigues(poses[i,:3,:3])
            ret[i,:3] = rvec.reshape(-1)
            ret[i,3:] = poses[i,:3,3]
        return ret
    else:
        raise 'Invalid Input'

def T6_to_T44(poses):
    if len(poses.shape) == 1:
        ret = np.zeros((4,4),poses.dtype)
        R,_ = cv2.Rodrigues(poses[:3])
        ret[:3,:3] = R
        ret[:3,3] = poses[3:6]
        ret[3,3] = 1
        return ret
    elif len(poses.shape) == 2:
        N = poses.shape[0]
        ret = np.zeros((N,4,4),poses.dtype)
        for i in range(N):
            R,_ = cv2.Rodrigues(poses[i,:3])
            ret[i,:3,:3] = R
            ret[i,:3,3] = poses[i,3:6]
            ret[i,3,3] = 1
        return ret
    else:
        raise 'Invalid Input'


