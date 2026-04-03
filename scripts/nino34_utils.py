import numpy as np


def get_nino34_indices_and_weights(lat_grid: np.ndarray, lon_grid: np.ndarray):
    lat_grid = np.asarray(lat_grid).squeeze()
    lon_grid = np.asarray(lon_grid).squeeze()

    if lon_grid.ndim != 1 or lat_grid.ndim != 1:
        raise ValueError(f"lat/lon must be 1D arrays, got lat={lat_grid.shape}, lon={lon_grid.shape}")

    if np.nanmax(lon_grid) > 180:
        lon_left, lon_right = 190.0, 240.0
    else:
        lon_left, lon_right = -170.0, -120.0

    lat_idx = np.where((lat_grid <= 5.0) & (lat_grid >= -5.0))[0]
    lon_idx = np.where((lon_grid >= lon_left) & (lon_grid <= lon_right))[0]

    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError("Nino3.4 region indices are empty; check grid coordinates and units.")

    weights = np.cos(np.deg2rad(lat_grid[lat_idx]))[:, np.newaxis]
    return lat_idx, lon_idx, weights


def nino34_weighted_mean(field2d: np.ndarray, lat_idx: np.ndarray, lon_idx: np.ndarray, weights: np.ndarray):
    region = field2d[lat_idx.min():lat_idx.max() + 1, lon_idx.min():lon_idx.max() + 1]
    valid = ~np.isnan(region)
    if not valid.any():
        return np.nan

    w = np.broadcast_to(weights, region.shape)
    return float(np.average(region[valid], weights=w[valid]))
