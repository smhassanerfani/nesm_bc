import os
import json
import warnings
from collections import defaultdict
import numpy as np
import xesmf as xe
import xarray as xr
from tqdm import tqdm
from dask.diagnostics import ProgressBar

# Assume these are available in your environment
from utils import vertical_aggregation, get_area_for_latlongrid, modelE_regridder, optimize_zarr


def get_dataset_for_timestamp(file_group):
    """
    Loads and combines datasets for a single timestamp lazily (Dask-backed).
    """
    # Load datasets lazily using xarray
    ds1 = xr.open_dataset(file_group['aijh1E3oma'])
    ds2 = xr.open_dataset(file_group['aijlh1E3oma'])
    ds4 = xr.open_dataset(file_group['taijlh1E3oma'])
    ds5 = xr.open_dataset(file_group['tNDaijh1E3oma'])

    ds = xr.Dataset()

    # Consolidate and assign variables
    ds['bcb'] = ds4.BCB
    ds['airmass'] = ds4.airmass
    ds['u'] = ds2.u
    ds['v'] = ds2.v
    ds['p'] = ds2.p_3d
    ds['z'] = ds2.z
    ds['t'] = ds2.t
    ds['q'] = ds2.q
    ds['th'] = ds2.th
    ds['bcb_src'] = ds5.BCB_biomass_src
    ds['pblh'] = ds1.pblht_bp
    ds['cell_area'] = ds1.axyp  # lat x lon

    # Convert extensive variables (kg/m^2 to total kg)
    ds["airmass_total"] = ds['airmass'] * ds['cell_area'] 
    ds.airmass_total.attrs = {"units": "kg", "long_name": "Total air mass per grid cell"}

    ds["bcb_total"] = ds['bcb'] * 1E-11 * ds["airmass_total"]
    ds.bcb_total.attrs = {"units": "kg/s", "long_name": "Total BCB mass per grid cell"}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Converting a CFTimeIndex with dates from a non-standard calendar")
        # Ensure 'time' variable is converted to a standard datetime
        ds['time'] = ds.indexes['time'].to_datetimeindex(time_unit='us')
        
    return ds.astype(np.float32)  # Ensure consistent dtype for Dask operations


def process_and_regrid_timestamp(ds, area, regridders, l15_interp):
    """
    Applies regridding and vertical aggregation using pre-computed regridders.
    This function contains all the computationally intensive, lazy operations.
    """
    regridder_extensive = regridders['extensive']
    regridder_intensive = regridders['intensive']
    
    # 1. Extensive Regridding (Conservative)
    ds_regrid = xr.Dataset()
    
    for var in ["airmass_total", "bcb_total"]:
        # Divide by source cell area, regrid, multiply by target area
        tmp = regridder_extensive(ds[var] / ds['cell_area'], keep_attrs=True) * area
        
        # Normalize to conserve global sum 
        ds_regrid[var] = (
            tmp / tmp.sum(["lat", "lon"]) * ds[var].sum(["lat", "lon"])
        )

    # Calculate densities (kg/m²)
    bcb_density = ds_regrid['bcb_total'] / area
    airmass_density = ds_regrid['airmass_total'] / area

    # Convert from mass density back to mixing ratio
    ds_regrid['bcb'] = xr.where(
        airmass_density > 0, bcb_density / airmass_density / 1E-11, 0
    )

    ds_regrid.bcb.attrs = {"units": "10^-11 kg / kg air", "long_name": "BCB mixing ratio"}
    
    ds_regrid['airmass'] = airmass_density
    ds_regrid.airmass.attrs = {"units": "kg m-2", "long_name": "air mass density"}

    # 2. Intensive Regridding (Bilinear)
    intensive_vars =['u', 'v', 'p', 'z', 't', 'th', 'q', 'pblh']
    # Select variables from the original dataset for regridding
    ds_intensive = ds[intensive_vars]
    
    regridded_intensive_vars = regridder_intensive(ds_intensive, keep_attrs=True)

    for var in intensive_vars:
        ds_regrid[var] = regridded_intensive_vars[var]
        ds_regrid[var].attrs = ds[var].attrs
        
    # 3. BCB Source Regridding (Conservative)
    ds_regrid['bcb_src'] = regridder_extensive(ds['bcb_src'], keep_attrs=True)
    ds_regrid.bcb_src.attrs = ds['bcb_src'].attrs

    # Final cleanup and vertical aggregation
    ds_regrid["cell_area"] = area
    ds_regrid = ds_regrid.drop_vars(['airmass_total', 'bcb_total'])

    ds_vregrid = vertical_aggregation(ds_regrid, levels=l15_interp)

    return ds_vregrid.astype(np.float32)

# ------------------------------------------------------------------------------
#                                MAIN FUNCTION
# ------------------------------------------------------------------------------

def main(dataset_name, save_dir):

    # --- 1. Initialization (Run Once) ---
    with open('plm.json', 'r') as jf:
        plm = json.load(jf)['pressureLevels_hPa']

    # Your fixed vertical aggregation scheme
    l15_interp = [
        [0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], 
        [15, 16, 17], [18], [19], [20], [21], [22, 23], 
        list(range(24, 30)), list(range(30, 35)), list(range(35, 62)),
    ]

    # Your fixed target grid
    new_grid = xr.Dataset(
        coords={
            "lat": np.linspace(-90 + 90 / 64, 90 - 90 / 64, 64),
            "lon": np.linspace(-180 + (180 / 128), 180 - (180 / 128), 128)
        }
    )
    area = get_area_for_latlongrid(coords=new_grid.coords) 
    
    # --- 2. File Grouping ---
    file_groups = defaultdict(dict)
    for root, dirs, files in os.walk(f'/home/serfani/serfani_data1/{dataset_name}'):
        sorted_files = sorted([f for f in files if f.endswith('.nc')])
        for filename in sorted_files:
            timestamp = filename.split('.')[0]
            file_type = filename.split('.')[1]
            if file_type in ['aijh1E3oma', 'aijlh1E3oma', 'taijlh1E3oma', 'tNDaijh1E3oma']:
                file_groups[timestamp][file_type] = os.path.join(root, filename)

    required_types = ['aijh1E3oma', 'aijlh1E3oma', 'taijlh1E3oma', 'tNDaijh1E3oma']
    valid_groups = {
        ts: fg for ts, fg in file_groups.items() 
        if all(ftype in fg for ftype in required_types)
    }

    if not valid_groups:
        print("No complete file groups found. Exiting.")
        return
    
    regridders = {
        'extensive': modelE_regridder(new_grid, intensive=False, periodic=True),
        'intensive': modelE_regridder(new_grid, intensive=True, periodic=True)
    }

    # --- 4. Parallel Processing (Lazy Dask Tasks) ---

    print("Setting up Dask computation graph...")
    for timestamp in tqdm(valid_groups.keys(), desc="Creating Dask tasks"):
        if int(timestamp) > 18501231:

            file_group = valid_groups[timestamp]
            
            # ds is a Dask-backed Dataset (assuming get_dataset_for_timestamp loads lazily)
            ds = get_dataset_for_timestamp(file_group)
            rds = process_and_regrid_timestamp(ds, area, regridders, l15_interp)

            # 1. Modify chunk_dict to use time=48
            # This sets the Dask array chunk size *before* saving.
            chunk_dict = dict(time=-1, lat=64, lon=128, level=15) 
            rds = rds.chunk(chunk_dict)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part in the Zarr format 3 specification")
            
                if not os.path.exists(save_dir):
                    
                    rds.to_zarr(
                        save_dir, 
                        mode='w', 
                        consolidated=True, 
                        compute=True, 
                    )
                else:
                    rds.to_zarr(save_dir, mode="a", append_dim="time", consolidated=True)



if __name__ == "__main__":
    dataset_name = 'E3OMA1850'
    save_dir = f'/home/serfani/serfani_data1/{dataset_name}-R64x128V15.zarr'
    # main(dataset_name, save_dir)
    
    ds = xr.open_zarr(save_dir, consolidated=True)
    ds_opt = optimize_zarr(ds)
    
    with ProgressBar():
        ds_opt.to_zarr(save_dir.replace('.zarr', '-opt.zarr'), mode='w', consolidated=True, compute=True)
