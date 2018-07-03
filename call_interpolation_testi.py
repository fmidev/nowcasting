import interpolate_and_verify
import h5py
import hiisi
import numpy as np
import argparse
import datetime
import ConfigParser
import matplotlib.pyplot as plt
import netCDF4

def read_nc(image_nc_file):
    tempds = netCDF4.Dataset(image_nc_file)
    internal_variable = tempds.variables.keys()[-1]
    temps = np.array(tempds.variables[internal_variable][:]) # This picks the actual data
    nodata = tempds.variables[internal_variable].missing_value
    time_var = tempds.variables["time"]
    dtime = netCDF4.num2date(time_var[:],time_var.units) # This produces an array of datetime.datetime values
    
    # Outside of area all the values are missing. Leave them as they are. They're not affected by the motion vector calculations
    mask_nodata = np.ma.masked_where(temps == nodata,temps)
    # Pick min/max values from the data
    temps_min= temps[np.where(~np.ma.getmask(mask_nodata))].min()
    temps_max= temps[np.where(~np.ma.getmask(mask_nodata))].max()

    # The script returns four variables: the actual data, timestamps, nodata_mask and the actual nodata value
    return temps, temps_min, temps_max, dtime, mask_nodata, nodata
 




def read_HDF5(image_h5_file):
    #Read RATE or DBZH from hdf5 file
    print 'Extracting data from image h5 file'
    comp = hiisi.OdimCOMP(image_h5_file, 'r')
    #Read RATE array if found in dataset      
    test=comp.select_dataset('RATE')
    if test != None:
        image_array=comp.dataset
        quantity='RATE'
    else:
        #Look for DBZH array
        test=comp.select_dataset('DBZH')
        if test != None:
            image_array=comp.dataset
            quantity='DBZH'
        else:
            print 'Error: RATE or DBZH array not found in the input image file!'
            sys.exit(1)
    if quantity == 'RATE':
        quantity_min = options.R_min
        quantity_max = options.R_max
    if quantity == 'DBZH':
        quantity_min = options.DBZH_min
        quantity_max = options.DBZH_max
    
   

    #Read nodata and undetect values from metadata for masking
    gen = comp.attr_gen('nodata')
    pair = gen.next()
    nodata = pair.value
    gen = comp.attr_gen('undetect')
    pair = gen.next()
    undetect = pair.value
    #Read gain and offset values from metadata
    gen = comp.attr_gen('gain')
    pair = gen.next()
    gain = pair.value
    gen = comp.attr_gen('offset')
    pair = gen.next()
    offset = pair.value
    #Read timestamp from metadata
    gen = comp.attr_gen('date')
    pair = gen.next()
    date = pair.value
    gen = comp.attr_gen('time')
    pair = gen.next()
    time = pair.value

    timestamp=date+time

    #Masks of nodata and undetect
    mask_nodata = np.ma.masked_where(image_array == nodata,image_array)
    mask_undetect = np.ma.masked_where(image_array == undetect,image_array)

    #Change to physical values
    image_array=image_array*gain+offset

    #Mask nodata and undetect values to zero
    image_array[np.where(np.logical_or( np.ma.getmask(mask_nodata), np.ma.getmask(mask_undetect) ))] = 0

    return image_array, quantity, quantity_min, quantity_max, timestamp, mask_nodata



def farneback_params_config():
    config = ConfigParser.RawConfigParser()
    config.read("compute_advinterp.cfg")
    farneback_pyr_scale  = config.getfloat("optflow",    "pyr_scale")
    farneback_levels     = config.getint("optflow",      "levels")
    farneback_winsize    = config.getint("optflow",      "winsize")
    farneback_iterations = config.getint("optflow",      "iterations")
    farneback_poly_n     = config.getint("optflow",      "poly_n")
    farneback_poly_sigma = config.getfloat("optflow",    "poly_sigma")
    farneback_params = (farneback_pyr_scale,  farneback_levels, farneback_winsize, 
                        farneback_iterations, farneback_poly_n, farneback_poly_sigma, 0)
    return farneback_params


def init_filedict_interpolation(image_h5_file):

    #Write metadata to file_dict
    comp = hiisi.OdimCOMP(image_h5_file, 'r')
    file_dict_interp = {
        '/':dict(comp['/'].attrs.items()),
        '/how':dict(comp['/how'].attrs.items()),
        '/where':dict(comp['/where'].attrs.items())}
    file_dict_interp['/dataset1/data1/what'] = {
        'gain':0.01,
        'nodata':65535,
        'offset':0,
        'product':np.string_("COMP"),
        'quantity':np.string_("RATE"),
        'undetect':0}

    return file_dict_interp


def init_filedict_accumulation(image_h5_file):

    #Read metadata from image h5 file
    comp = hiisi.OdimCOMP(image_h5_file, 'r')

    #Write metadata to file_dict
    file_dict_accum = {
        '/':dict(comp['/'].attrs.items()),
        '/how':dict(comp['/how'].attrs.items()),
        '/where':dict(comp['/where'].attrs.items())}

    return file_dict_accum


def write_interpolated_h5(output_h5,interpolated_image,file_dict_interp,date,time):

    #Insert date and time of interpolated frame to file_dict
    file_dict_interp['/what'] = {
        'date':date,
        'object':np.string_("COMP"),
        'source':np.string_("ORG:247"),
        'time':time,
        'version':np.string_("H5rad 2.0")}

    #Insert interpolated dataset into file_dict
    file_dict_interp['/dataset1/data1/data'] = {
        'DATASET':interpolated_image,
        'COMPRESSION':'gzip',
        'COMPRESSION_OPTS':6,
        'CLASS':np.string_("IMAGE"),
        'IMAGE_VERSION':np.string_("1.2")}

    #Write hdf5 file from file_dict 
    with hiisi.HiisiHDF(output_h5, 'w') as h:
        h.create_from_filedict(file_dict_interp)


def write_accumulated_h5(output_h5,accumulated_image,file_dict_accum,date,time,startdate,starttime,enddate,endtime):

    #Insert date and time to file_dict
    file_dict_accum['/what'] = {
        'date':date,
        'object':np.string_("COMP"),
        'source':np.string_("ORG:247"),
        'time':time,
        'version':np.string_("H5rad 2.0")}
    #Insert startdate and -time and enddate- and time
    file_dict_accum['/dataset1/data1/what'] = {
        'gain':0.001,
        'nodata':65535,
        'offset':0,
        'product':np.string_("COMP"),
        'quantity':np.string_("ACRR"),
        'undetect':0,
        'startdate':startdate,
        'starttime':starttime,
        'enddate':enddate,
        'endtime':endtime}
    #Insert accumulated dataset into file_dict
    file_dict_accum['/dataset1/data1/data'] = {
        'DATASET':accumulated_image,
        'COMPRESSION':'gzip',
        'COMPRESSION_OPTS':6,
        'CLASS':np.string_("IMAGE"),
        'IMAGE_VERSION':np.string_("1.2")}
    #Write hdf5 file from file_dict 
    with hiisi.HiisiHDF(output_h5, 'w') as h:
        h.create_from_filedict(file_dict_accum)



def main():
    
    # FIRST LEARN HOW TO READ NETCDF OBS AND MODEL FIELDS IN. The function below is written only for reading HDF5 radar data. Also radar observations need to be in some kind of accumulation form as pal_skandinavia data are always 1h accumulations.

    # In the "verification mode", the idea is to load in the "observational" and "forecast" datasets as numpy arrays. Both of these numpy arrays ("image_array") contain ALL the timesteps contained also in the files themselves. In addition, the returned variables "timestamp" and "mask_nodata" contain the values for all the timesteps.
    # Variables ("quantity", "quantity_min", "quantity_max") are defined outside the data retrievals

    # First precipitation field is from Tuliset2/analysis. What about possible scaling???
    if options.parameter == 'Precipitation1h_TULISET': # This needs to be done better
        image_array1, quantity1_min, quantity1_max, timestamp1, mask_nodata1 = read_HDF5("/fmi/data/nowcasting/testdata_radar/opera_rate/T_PAAH21_C_EUOC_20180613120000.hdf")
        # image_array2, quantity2_min, quantity2_max, timestamp2, mask_nodata2 = read_HDF5("/fmi/data/nowcasting/testdata_radar/opera_rate/T_PAAH21_C_EUOC_20180613121500.hdf")
    else:
        image_array1, quantity1_min, quantity1_max, timestamp1, mask_nodata1, nodata1 = read_nc("/fmi/data/nowcasting/testdata/obsdata.nc")
        quantity1 = "options.parameter"
    # The second field is always a model forecast field
    image_array2, quantity2_min, quantity2_max, timestamp2, mask_nodata2, nodata2 = read_nc("/fmi/data/nowcasting/testdata/modeldata.nc")
    quantity2 = "options.parameter"

    # Missing_values of image_array2 ("nodata2") are changed to "nodata1". The larger data is made smaller so that the actual data points in the two data sets have the same geographical domain (LAPS data covers a slighlty larger domain than pal).
    if np.ma.count_masked(mask_nodata2) > np.ma.count_masked(mask_nodata1):
        image_array1[np.where( np.ma.getmask(mask_nodata2) )] = nodata1
        image_array2[np.where( np.ma.getmask(mask_nodata2) )] = nodata1
        mask_nodata1 = mask_nodata2
    else:
        image_array1[np.where( np.ma.getmask(mask_nodata1) )] = nodata1
        image_array2[np.where( np.ma.getmask(mask_nodata1) )] = nodata1
        mask_nodata2 = mask_nodata1
    
    # Defining definite min/max values from the two fields
    R_min=min(quantity1_min,quantity2_min)
    R_max=max(quantity1_max,quantity2_max)

    # # Mask nodata values to be just slightly less than the lowest field value
    # image_array1[np.where(np.ma.getmask(mask_nodata1))] = R_min
    # nodata = -nodata


    # DATA IS NOW LOADED AS A NORMAL NUMPY NDARRAY
    
    # #Calculate interpolated images
    # interpolated_images=interpolation.advection(obsfields=image_array1, modelfields=image_array2, mask_nodata=mask_nodata1, farneback_params=farneback_params, R_min=R_min, R_max=R_max, missingval=nodata1, logtrans=False, predictability=options.predictability, seconds_between_steps=options.seconds_between_steps)

    
    # raise ScriptExit( "A Good Reason" )
    # return;

    # #Read parameters from config file for interpolation (or optical flow algorithm, find out this later!). The function for reading the parameters is defined above.
    farneback_params=farneback_params_config()
    
    #Calculate number of interpolated frames from timestamps and minutes between steps
    formatted_time_first=datetime.datetime.strptime(timestamp1,'%Y%m%d%H%M%S')
    formatted_time_second=datetime.datetime.strptime(timestamp2,'%Y%m%d%H%M%S')
    timediff_seconds=(formatted_time_second-formatted_time_first).total_seconds()
    n_interp_frames=int(timediff_seconds/options.seconds_between_steps-1) # Here, int type (timediff_seconds) is divided with another int type (seconds_between_steps) so the result is int as well

    # R1 = image_array1
    # R2 = image_array2
    # R_min = quantity1_min
    # R_max = quantity1_max
    # logtrans=False
    # missingval=nan

    # #Calculate interpolated images
    # interpolated_images=interpolation.advection(image_array1, image_array2, n_interp_frames, farneback_params, quantity1_min, quantity1_max, logtrans=False)

    # #Init file_dicts
    # file_dict_interp=init_filedict_interpolation(options.obsdata)
    # file_dict_accum=init_filedict_accumulation(options.obsdata)

    # #Init 5 min sum
    # sum_5min=image_array1/(60*60/options.seconds_between_steps)
    # startdate=timestamp1[8:]
    # starttime=timestamp1[:8]
    # #Write interpolated images to separate hdf5 files
    # for n in range(n_interp_frames):
    #     print n
    #     #Calculate new timestamp
    #     seconds=(n+1)*options.seconds_between_steps
    #     new_time=formatted_time_first + datetime.timedelta(seconds=seconds)
    #     new_timestamp=new_time.strftime("%Y%m%d%H%M%S")
    #     #Next timestamp (for saving the 5 min sum)
    #     next_seconds=(n+2)*options.seconds_between_steps
    #     next_time=formatted_time_first + datetime.timedelta(seconds=next_seconds)
    #     next_timestamp=next_time.strftime("%Y%m%d%H%M%S")

    #     #Calculate and save 5 minute sum (mm/h -> mm/5 min)
    #     next_minutes=next_seconds/60.0
    #     sum_5min=sum_5min+interpolated_images[n]/(3600/options.seconds_between_steps)
    #     if next_minutes%5 == 0:
    #         #Convert 5 min sum to 16 byte unsigned integer
    #         sum_5min=np.uint16(sum_5min*1000)
    #         #Mask nodata values with 2^16-1
    #         sum_5min[np.where(np.ma.getmask(mask_nodata1))]=2**16-1
    #         #Write to file
    #         outfile_5min=options.output_sum.format(next_timestamp)
    #         date=next_timestamp[:8]
    #         time=next_timestamp[8:]
    #         enddate=date
    #         endtime=time
    #         write_accumulated_h5(outfile_5min,sum_5min,file_dict_accum,date,time,startdate,starttime,enddate,endtime)
    #         #Init next 5 min sum
    #         sum_5min=interpolated_images[n]/(60*60/options.seconds_between_steps)
    #         startdate=date
    #         starttime=time

    #     #Convert interpolated image to 16 byte unsigned integer
    #     interpolated_image=np.uint16(interpolated_images[n]*100)
        
    #     #Mask nodata values with 2^16-1
    #     interpolated_image[np.where(np.ma.getmask(mask_nodata1))] = 2**16-1

    #     #Write to file
    #     outfile=options.output_interpolate.format(new_timestamp)
    #     date=new_timestamp[:8]
    #     time=new_timestamp[8:]
    #     write_interpolated_h5(outfile,interpolated_image,file_dict_interp,date,time)
    #     print 'Wrote', outfile



    print(dir())
    
    


if __name__ == '__main__':

    #Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--obsdata',
                        help='Obs data, representing the first time step used in image morphing.')
    parser.add_argument('--modeldata',
                        help='Model data, representing the last time step used in image morphing.')
    parser.add_argument('--seconds_between_steps',
                        type=int,
                        default=3600,
                        help='Seconds between interpolated steps.')
    parser.add_argument('--R_min',
                        type=float,
                        default=0.1,
                        help='Minimum precipitation intensity for optical flow computations. Values below R_min are set to zero.')
    parser.add_argument('--R_max',
                        type=float,
                        default=30.0,
                        help='Maximum precipitation intensity for optical flow computations. Values above R_max are clamped.')
    parser.add_argument('--DBZH_min',
                        type=float,
                        default=10,
                        help='Minimum DBZH for optical flow computations. Values below DBZH_min are clamped.')
    parser.add_argument('--DBZH_max', 
                        type=float,
                        default=45,
                        help='Maximum DBZH for optical flow computations. Values above DBZH_max are clamped.')
    parser.add_argument('--output_interpolate',
                        default='test_{}.nc',
                        help='Output hdf5 file name, leave brackets {} for timestamp.')
    parser.add_argument('--predictability',
                        default='3',
                        help='Predictability in hours. Between the analysis and forecast of this length, forecasts need to be interpolated')
    parser.add_argument('--parameter',
                        default='Temperature',
                        help='Variable which is handled.')


    options = parser.parse_args()
    main()
