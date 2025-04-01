import numpy as np

def hip_referenced_normalization(keypoints):
    """Center keypoints around hips"""
    hips = (keypoints[:, 23] + keypoints[:, 24]) / 2
    return keypoints - hips[:, np.newaxis, :]

def anthropometric_normalization(keypoints):
    """Scale normalization using shoulder-hip distance"""
    shoulders = (keypoints[:, 11] + keypoints[:, 12]) / 2
    hips = (keypoints[:, 23] + keypoints[:, 24]) / 2
    scale = np.linalg.norm(shoulders - hips, axis=1)
    scale = np.where(scale == 0, 1e-6, scale)
    return keypoints / scale[:, np.newaxis, np.newaxis]