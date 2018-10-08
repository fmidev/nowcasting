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



def quiver(filename, flow_x, flow_y, quality, grid_spacing=20, arrow_scaling='default', arrow_width=0.004, dpi=100, arrow_color=(1., 0., 0., 0.5)):

    if flow_x.shape!= flow_y.shape:
        raise Exception('Shapes of x and y component arrays do not match')

    height, width = flow_x.shape

    # x, y grid indexes of flow arrows                                                                           
    y, x = np.mgrid[grid_spacing/2:height:grid_spacing, grid_spacing/2:width:grid_spacing]

    V = flow_y[::grid_spacing, ::grid_spacing]
    U = flow_x[::grid_spacing, ::grid_spacing]

    #Invert y vertically for plotting and wind v component changes to negative.                                                             
    y=np.flip(y, 0)
    V=-V

    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)

    if arrow_scaling == 'default':
        # Scaling: max value == 1                                                             
        N = np.sqrt(U**2 + V**2)
        max_magnitude = np.max(N)
        U2, V2 = U/max_magnitude, V/max_magnitude

    elif arrow_scaling == 'unitlength':
        # Scaling: all arrows = 1
        N = np.sqrt(U**2 + V**2)
        U2, V2 = U/N, V/N

    else:
        U2, V2 = U, V

    plt.quiver(x, y, U, V, width=arrow_width, color=arrow_color, scale=arrow_scaling)

    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')
    plt.tight_layout(pad=0.)
    plt.savefig(filename, transparent=True)


def main():

    temps, lats, lons, temps_min, temps_max, dtime, mask_nodata, nodata=read_nc(options.input_file)

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


if __name__ == '__main__':

    #Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        help='Input netcdf file to plot.')
    parser.add_argument('--output_file',
                        help='Output png file.')

    options = parser.parse_args()
    main()
