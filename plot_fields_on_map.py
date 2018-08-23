import netCDF4
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pyresample import geometry
from pyproj import Proj


def read_nc(image_nc_file):
    tempds = netCDF4.Dataset(image_nc_file)
    internal_variable = tempds.variables.keys()[-1]
    temps = np.array(tempds.variables[internal_variable][:]) # This picks the actual data
    lats = np.array(tempds.variables['lat'][:]) # This picks the actual data
    lons = np.array(tempds.variables['lon'][:]) # This picks the actual data

    nodata = tempds.variables[internal_variable].missing_value
    time_var = tempds.variables["time"]
    dtime = netCDF4.num2date(time_var[:],time_var.units) # This produces an array of datetime.datetime values
    
    # Outside of area all the values are missing. Leave them as they are. They're not affected by the motion vector calculations
    mask_nodata = np.ma.masked_where(temps == nodata,temps)
    # Pick min/max values from the data
    temps_min= temps[np.where(~np.ma.getmask(mask_nodata))].min()
    temps_max= temps[np.where(~np.ma.getmask(mask_nodata))].max()

    # The script returns four variables: the actual data, timestamps, nodata_mask and the actual nodata value
    return temps, lats, lons, temps_min, temps_max, dtime, mask_nodata, nodata


def resample_projection(lats, lons):

    #Grid definition information of existing data
    grid_def = geometry.GridDefinition(lons=lons, lats=lats)

    #Wanted projection
    area_id = 'laps_scan'
    description = 'LAPS Scandinavian domain'
    proj_id = 'laps_scan'
    proj='stere'
    proj='epsg:3995'
    lon_0=20.0
    lat_0=90.0

    #x_size = 1000
    #y_size = 1000

    #Corner points to be converted to wanted projection
    lon1=-2.448425
    lat1=68.79139
    lon2=29.38635
    lat2=54.67893

    #Calculate coordinate points in projection
    p = Proj(init=proj)
    x1, y1 = p(lon1,lat1)
    x2, y2 = p(lon2,lat2)

    print x1,y1, x2,y2

    print abs(x1-x2)/abs(y1-y2)
    print abs(y1-y2)/abs(x1-x2)



    area_extent = (x1,y1,x2,y2)
    proj_dict = {'a': '6371228.0', 'units': 'm', 'lon_0': lon_0, 'proj': proj, 'lat_0': lat_0}
    area_def = geometry.AreaDefinition(area_id, description, proj_id, proj_dict, x_size, y_size, area_extent)

    print area_def

    #Finland domain
    #lon1=16.52893
    #lat1=70.34990
    #lon2=31.85138
    #lat2=58.76623

        
    #return m, x, y


def plot_imshow(temps,vmin,vmax):

    plt.imshow(temps[0,:,:],vmin=vmin,vmax=vmax)
    plt.show()


def main():

    print options.input_file
    print options.output_file

    temps, lats, lons, temps_min, temps_max, dtime, mask_nodata, nodata=read_nc(options.input_file)

    #Gridded lat and lon
    lons, lats=np.meshgrid(lons, lats)
    plot_imshow(temps,temps_min,temps_max)

    resample_projection(lats, lons)

    




if __name__ == '__main__':

    #Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        help='Input netcdf file to plot.')
    parser.add_argument('--output_file',
                        help='Output png file.')

    options = parser.parse_args()
    main()
