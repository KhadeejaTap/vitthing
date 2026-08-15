import numpy as np


SENSOR_ZMIN = 300.0
SENSOR_ZMAX = 8333.0


def compute_log_params(zmin=SENSOR_ZMIN, zmax=SENSOR_ZMAX):
    """fixed sensor range -> fixed alpha, beta. same every frame."""
    alpha = np.log(zmax) - np.log(zmin)
    beta = np.log(zmin)
    return float(alpha), float(beta)


def normalize_depth(depth_filled, alpha, beta):
    """log-normalize filled depth to ~[0,1] using given alpha/beta. eq(1)."""
    return (np.log(depth_filled) - beta) / alpha


def build_input_tensor(depth_filled, valid_mask, alpha, beta):
    """
    returns 3ch tensor (3,H,W) float32, scaled to [-1,1]:
    ch0/ch1 = normalized depth duplicated
    ch2 = validity mask (sparse, real measurements only)
    """
    zhat = normalize_depth(depth_filled, alpha, beta)
    mask = valid_mask.astype(np.float32)

    tensor = np.stack([zhat, zhat, mask], axis=0).astype(np.float32)
    tensor = tensor * 2.0 - 1.0  # [0,1] -> [-1,1]
    return tensor


def denormalize_depth(dhat, alpha, beta):
    """eq(3): recover metric depth from network output."""
    return np.exp(alpha * dhat + beta)
