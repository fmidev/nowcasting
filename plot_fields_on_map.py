# -*- coding: utf-8 -*-                                                                                                                 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import netCDF4
import argparse
import numpy as np
from pyresample import geometry, image
from pyproj import Proj
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter
import ConfigParser


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


def resample_data(lats, lons, data):

    #Grid definition information of existing data
    grid_def = geometry.GridDefinition(lons=lons, lats=lats)

    #Wanted projection
    area_id = 'laps_scan'
    description = 'LAPS Scandinavian domain'
    proj_id = 'stere'
    proj = 'epsg:3995'
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

    x_size=1000
    y_size=abs(x1-x2)/abs(y1-y2)*1000

    area_extent = (x1,y1,x2,y2)
    proj_dict = {'a': '6371228.0', 'units': 'm', 'lon_0': lon_0, 'proj': proj_id, 'lat_0': lat_0}
    area_def = geometry.AreaDefinition(area_id, description, proj_id, proj_dict, x_size, y_size, area_extent)

    print area_def

    #Finland domain
    #lon1=16.52893
    #lat1=70.34990
    #lon2=31.85138
    #lat2=58.76623


    #Resampling data
    laps_con_quick = image.ImageContainerQuick(data, grid_def)
    area_con_quick = laps_con_quick.resample(area_def)
    result_data_quick = area_con_quick.image_data
    laps_con_nn = image.ImageContainerNearest(data, grid_def, radius_of_influence=50000)
    area_con_nn = laps_con_nn.resample(area_def)
    result_data_nn = area_con_nn.image_data

    print result_data_nn





        
    #return m, x, y


def plot_imshow(temps,vmin,vmax,outfile):

    plt.imshow(temps,cmap='jet',vmin=vmin,vmax=vmax,origin="lower")
    #plt.colorbar()
    plt.axis('off')
    plt.tight_layout(pad=0.)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(outfile,bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_only_colorbar(vmin,vmax,units,outfile,cmap):

    fig = plt.figure(figsize=(8, 1))
    ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
    #cmap = matplotlib.cm.cmap
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cb1 = matplotlib.colorbar.ColorbarBase(ax1,cmap='jet', norm=norm,orientation='horizontal')
    cb1.set_label(units)
    plt.savefig(outfile,bbox_inches='tight')
    plt.show()



def main():

    temps, lats, lons, temps_min, temps_max, dtime, mask_nodata, nodata=read_nc(options.input_file)

    #Gridded lat and lon
    #lons, lats=np.meshgrid(lons, lats)

    #Read config parameters from config file
    parameter="Temperature"
    config_file="color_settings.cfg"
    config = ConfigParser.RawConfigParser()
    config.read(config_file)
    vmin = config.getfloat(parameter, "vmin")
    vmax = config.getfloat(parameter, "vmax")
    units = config.get(parameter, "units")
    cmap = config.get(parameter, "cmap")

    #585 553 585 553

    print lats[0,0], lons[0,0]
    print lats[0,552], lons[0,552]
    print lats[584,0], lons[584,0]
    print lats[584,552], lons[584,552]
    


    print vmin, vmax, units, cmap

    print temps.shape
    pal_shape=(7, 236, 222)
    pal_shape_2=(236, 222)
    new_temps=np.zeros(pal_shape)
    for n in range(0,temps.shape[0]):
        #Resize
        new_temps[n]=imresize(temps[n], pal_shape_2, interp='bilinear', mode='F')
        #Gaussian filter to blur LAPS data
        new_temps[n]=gaussian_filter(new_temps[n], 1)
        filename=options.output_file+str(n)+'.png'
        print new_temps[n].shape
        plot_imshow(new_temps[n],temps_min,temps_max,filename)


    outfile='colorbar_' + parameter + '.png'
    plot_only_colorbar(temps_min,temps_max,units,outfile,cmap)

#    resample_data(lats, lons, temps)


if __name__ == '__main__':

    #Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        help='Input netcdf file to plot.')
    parser.add_argument('--output_file',
                        help='Output png file.')

    options = parser.parse_args()
    main()
