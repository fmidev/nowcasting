import netCDF4
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pyresample


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



def map_coord(lats, lons):

    #LAPS
    proj='stere'

    #Scandinavia domain
    lon1=-2.448425
    lat1=68.79139
    lon2=29.38635
    lat2=54.67893

    #Finland domain
    #lon1=16.52893
    #lat1=70.34990
    #lon2=31.85138
    #lat2=58.76623

    lon_0=20.0
    lat_0=90.0
    
    # create projection
    m = Basemap(llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2,\
            rsphere=6371000.,\
            resolution='l',projection=proj,\
            lat_1=lat_0,lat_2=lat_0,lat_0=lat_0,lon_0=lon_0)

    # Transform coordinates to figure coordinates
    x,y = m(lons,lats)
    
    return m, x, y




def plot_imshow(temps,vmin,vmax):

    plt.imshow(temps[0,:,:],vmin=vmin,vmax=vmax)
    plt.show()



def plot_map_proj(m,x,y,temps):

    m.drawcoastlines()
    m.drawparallels(np.arange(-80.,81.,20.))
    m.drawmeridians(np.arange(-180.,181.,20.))
    m.contour(x,y,temps)

    plt.title("North Polar Stereographic Projection")
    plt.show()




def main():

    print options.input_file
    print options.output_file

    temps, lats, lons, temps_min, temps_max, dtime, mask_nodata, nodata=read_nc(options.input_file)

    #Gridded lat and lon
    lons, lats=np.meshgrid(lons, lats)

    plot_imshow(temps,temps_min,temps_max)

    #Projection coordinates
    m, x, y=map_coord(lats, lons)
    plot_map_proj(m,x,y,temps)




if __name__ == '__main__':

    #Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        help='Input netcdf file to plot.')
    parser.add_argument('--output_file',
                        help='Output png file.')

    options = parser.parse_args()
    main()
