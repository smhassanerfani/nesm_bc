import numpy as np
import xesmf as xe
import xarray as xr

def get_area_for_latlongrid(coords=None):

    latstep = np.abs(np.diff(coords["lat"])).mean()
    lonstep = np.abs(np.diff(coords["lon"])).mean()
    area = xr.DataArray(
        np.stack(
            len(coords["lon"])
            * [
                np.pi
                * 6.3781E6**2
                * np.abs(
                    np.sin(np.radians(coords["lat"] + latstep / 2))
                    - np.sin(np.radians(coords["lat"] - latstep / 2))
                )
                * lonstep
                / 180
            ],
            axis=-1,
        ),
        coords={"lat": coords["lat"], "lon": coords["lon"]},
        dims=("lat", "lon"),
    )

    return area


def vertical_aggregation(ds, levels):

    vertical_ds = ds[[v for v in ds if "level" in ds[v].dims]]
    all_aggregated_ds = []
    
    for i, level in enumerate(levels):

        if len(level) > 1:
            # Get pressures for the levels to aggregate
            pressures = vertical_ds.p.isel(level=level)
            
            # Estimate pressure thickness from spacing between mid-layer pressures
            # For interior layers: use half the distance to each neighbor
            # For edge layers: use full distance to the single neighbor
            pressure_thickness = []
            
            for j in range(len(level)):
                if j == 0:  # First layer
                    if len(level) > 1:
                        # Use half distance to next layer, doubled
                        thickness = abs(pressures.isel(level=1) - pressures.isel(level=0))
                    else:
                        thickness = pressures.isel(level=0)  # Fallback for single layer
                elif j == len(level) - 1:  # Last layer
                    # Use half distance to previous layer, doubled
                    thickness = abs(pressures.isel(level=j-1) - pressures.isel(level=j))
                else:  # Interior layers
                    # Use half distance to each neighbor
                    thickness = (abs(pressures.isel(level=j-1) - pressures.isel(level=j)) + 
                               abs(pressures.isel(level=j) - pressures.isel(level=j+1))) / 2
                
                pressure_thickness.append(thickness)
            
            pressure_thickness = xr.concat(pressure_thickness, dim='level')
            pressure_weights = pressure_thickness / pressure_thickness.sum("level")

            ds_aggregated = (vertical_ds.isel(level=level) * pressure_weights).sum("level")

            ds_aggregated["airmass"] = vertical_ds.airmass.isel(level=level).sum("level")
            ds_aggregated["bcb"] = (
                vertical_ds.bcb * vertical_ds.airmass
            ).isel(level=level).sum("level") / ds_aggregated.airmass

            ds_aggregated = ds_aggregated.assign_coords(dict(level=[i]))
        else:
            ds_aggregated = vertical_ds.isel(level=level).assign_coords(dict(level=[i]))

        all_aggregated_ds.append(ds_aggregated)

    ds_aggregated = xr.concat(all_aggregated_ds, "level", data_vars='all')

    for v in ds:
        if "level" not in ds[v].dims:
            ds_aggregated[v] = ds[v]

    return ds_aggregated


def create_new_pressure_levels(level_groups, original_pressure_levels):
    """
    Create new pressure levels by combining original pressure levels based on the specified mapping.
    
    Parameters:
    - original_pressure_levels: List or array of the original pressure level values in hPa
    
    Returns:
    - new_pressure_levels: List of the new combined pressure levels in hPa
    - mapping: Dictionary mapping new pressure level indices to lists of original indices
    """
    
    # Calculate new pressure levels by averaging the original levels in each group
    new_pressure_levels = []
    for level_group in level_groups:
        if level_group:  # Check if the group is not empty
            # Extract the pressure values for this group
            group_pressures = [original_pressure_levels[idx] for idx in level_group]
            # Calculate the average pressure for this group
            avg_pressure = sum(group_pressures) / len(group_pressures)
            new_pressure_levels.append(round(avg_pressure, 1))
    
    # Create a mapping dictionary for reference
    mapping = {i: group for i, group in enumerate(level_groups)}
    
    return new_pressure_levels, mapping



def modelE_regridder(new_grid, intensive=True, periodic=True):
    """
    Create a regridder for the given dataset and target grid.
    
    Parameters:
    - ds: xarray Dataset containing the source data
    - new_grid: xarray Dataset defining the target grid with 'lat' and 'lon' coordinates
    - intensive: Boolean indicating whether to use bilinear (True) or conservative (False) regridding
    - periodic: Boolean indicating whether the longitude is periodic

    Returns:
    - regridder: xesmf Regridder object
    """

    ds = xr.open_dataset('/home/serfani/serfani_data1/E3OMA1850/18500101.aijh1E3oma.nc')

    method = 'bilinear' if intensive else 'conservative'

    regridder = xe.Regridder(ds, new_grid, method, periodic=periodic)

    return regridder


def optimize_zarr(ds):

    ds_2d = (
        ds[[v for v in ds.data_vars if "level" not in ds[v].dims]]
        .to_array("var2d")
        .transpose("time", "var2d", "lat", "lon")
        .astype("float32")
    )

    ds_2d = ds_2d.chunk(dict(time=48, var2d=-1, lat=-1, lon=-1))

    ds_3d = (
        ds[[v for v in ds.data_vars if "level" in ds[v].dims]]
        .to_array("var3d")
        .transpose("time", "var3d", "level", "lat", "lon")
        .astype("float32")
    )

    ds_3d = ds_3d.chunk(
        dict(time=48, var3d=-1, lat=-1, lon=-1, level=-1)
    )

    ds_opt = xr.Dataset({"variables_2d": ds_2d, "variables_3d": ds_3d})

    return ds_opt