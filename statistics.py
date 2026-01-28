import json
import numpy as np
import xarray as xr
from tqdm import tqdm

def compute_statistics(ds, variables):
    """
    Compute mean, std, min, max for specified variables in the dataset.
    For 3D variables (with level), compute statistics for each level separately.
    For 2D variables (no level), compute single statistics.

    Parameters:
    - ds: xarray.Dataset containing the data.
    - variables: list of variable names to compute statistics for.

    Returns:
    - stats: dict with statistics for each variable.
    """

    stats = {}
    weights = np.cos(np.deg2rad(ds.lat))

    for var in tqdm(variables, desc="Computing statistics"):
        if var not in ds:
            continue
        
        data = ds[var]
        
        # Check if variable has level dimension
        if 'level' in data.dims:
            # 3D variable: compute statistics for each level separately
            weighted_mean = data.weighted(weights).mean(dim=("lat", "lon", "time"))
            weighted_std = data.weighted(weights).std(dim=("lat", "lon", "time"))
            
            stats[var] = {
                'mean': weighted_mean.values.tolist(),  # Convert to list for each level
                'std': weighted_std.values.tolist(),    # Convert to list for each level
                'min': data.min(dim=("lat", "lon", "time")).values.tolist(),
                'max': data.max(dim=("lat", "lon", "time")).values.tolist()
            }
        else:
            # 2D variable: compute single statistics across lat, lon, time
            weighted_mean = data.weighted(weights).mean(dim=("lat", "lon", "time"))
            weighted_std = data.weighted(weights).std(dim=("lat", "lon", "time"))
            
            stats[var] = {
                'mean': weighted_mean.values.item(),  # Single value
                'std': weighted_std.values.item(),    # Single value
                'min': data.min().values.item(),
                'max': data.max().values.item()
            }

    tendency = ds.bcb.diff('time')
    stats['bcb_tendency'] = {
        'mean': tendency.weighted(weights).mean(dim=("lat", "lon", "time")).values.tolist(),
        'std': tendency.weighted(weights).std(dim=("lat", "lon", "time")).values.tolist(),
        'min': tendency.min(dim=("lat", "lon", "time")).values.tolist(),
        'max': tendency.max(dim=("lat", "lon", "time")).values.tolist()
    }

    return stats


def main():
    # Load a sample dataset (replace with actual data loading)
    ds = xr.open_dataset('/home/serfani/serfani_data1/E3OMA1850-R64x128V15.zarr', engine='zarr')
    ds = ds.chunk({'time': -1, 'lat': 64, 'lon': 128, 'level': 15})

    # Define variables to compute statistics for
    variables = ['u', 'v', 'p', 'z', 't', 'th', 'q', 'bcb', 'bcb_src', 'pblh']

    # Compute statistics
    stats = compute_statistics(ds, variables)

    # Print or save the statistics
    for var, stat in stats.items():
        print(f"Statistics for {var}:")
        if isinstance(stat['mean'], list):
            print(f"  Levels: {len(stat['mean'])}")
            for i, (mean_val, std_val) in enumerate(zip(stat['mean'], stat['std'])):
                print(f"  Level {i}: mean={mean_val:.4f}, std={std_val:.4f}")
        else:
            print(f"  mean={stat['mean']:.4f}, std={stat['std']:.4f}")
        print()

    with open('variable_statistics.json', 'w') as jf:
        json.dump(stats, jf, indent=4)


if __name__ == "__main__":
    main()