import interpolation
import h5py
import hiisi
import numpy as np
import argparse
import datetime
import ConfigParser

#Call interpolation between two radar composite fields.
#Calculate 5 minute accumulated rain. 
#Tuuli.Perttula@fmi.fi

def read_image(image_h5_file):
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

    return image_array, quantity, timestamp, mask_nodata


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
    

    # FIRST LEARN HOW TO READ NETCDF OBS AND MODEL FIELDS IN
    
    #Read image array hdf5's
    first_image_array, quantity, first_timestamp, mask_nodata = read_image(options.first_precip_field)
    second_image_array, quantity_second, second_timestamp, mask_nodata_second = read_image(options.second_precip_field)

    #Min and max values according to quantity (RATE/DBZH)
    if quantity == 'RATE':
        quantity_min = options.R_min
        quantity_max = options.R_max
    if quantity == 'DBZH':
        quantity_min = options.DBZH_min
        quantity_max = options.DBZH_max
        print 'Quantity in input hdf5 file is DBZH. RATE needed.'
        exit(1)

    #Read parameters from config file for interpolation
    farneback_params=farneback_params_config()

    #Calculate number of interpolated frames from timestamps and
    #minutes between steps
    formatted_time_first=datetime.datetime.strptime(first_timestamp,'%Y%m%d%H%M%S')
    formatted_time_second=datetime.datetime.strptime(second_timestamp,'%Y%m%d%H%M%S')
    timediff_seconds=(formatted_time_second-formatted_time_first).total_seconds()
    n_interp_frames=int(timediff_seconds/options.seconds_between_steps-1)

    #Calculate interpolated images
    interpolated_images=interpolation.advection(first_image_array, second_image_array, n_interp_frames, farneback_params, quantity_min, quantity_max, logtrans=False)

    #Init file_dicts
    file_dict_interp=init_filedict_interpolation(options.first_precip_field)
    file_dict_accum=init_filedict_accumulation(options.first_precip_field)

    #Init 5 min sum
    sum_5min=first_image_array/(60*60/options.seconds_between_steps)
    startdate=first_timestamp[8:]
    starttime=first_timestamp[:8]
    #Write interpolated images to separate hdf5 files
    for n in range(n_interp_frames):
        print n
        #Calculate new timestamp
        seconds=(n+1)*options.seconds_between_steps
        new_time=formatted_time_first + datetime.timedelta(seconds=seconds)
        new_timestamp=new_time.strftime("%Y%m%d%H%M%S")
        #Next timestamp (for saving the 5 min sum)
        next_seconds=(n+2)*options.seconds_between_steps
        next_time=formatted_time_first + datetime.timedelta(seconds=next_seconds)
        next_timestamp=next_time.strftime("%Y%m%d%H%M%S")

        #Calculate and save 5 minute sum (mm/h -> mm/5 min)
        next_minutes=next_seconds/60.0
        sum_5min=sum_5min+interpolated_images[n]/(3600/options.seconds_between_steps)
        if next_minutes%5 == 0:
            #Convert 5 min sum to 16 byte unsigned integer
            sum_5min=np.uint16(sum_5min*1000)
            #Mask nodata values with 2^16-1
            sum_5min[np.where(np.ma.getmask(mask_nodata))]=2**16-1
            #Write to file
            outfile_5min=options.output_sum.format(next_timestamp)
            date=next_timestamp[:8]
            time=next_timestamp[8:]
            enddate=date
            endtime=time
            write_accumulated_h5(outfile_5min,sum_5min,file_dict_accum,date,time,startdate,starttime,enddate,endtime)
            #Init next 5 min sum
            sum_5min=interpolated_images[n]/(60*60/options.seconds_between_steps)
            startdate=date
            starttime=time

        #Convert interpolated image to 16 byte unsigned integer
        interpolated_image=np.uint16(interpolated_images[n]*100)
        
        #Mask nodata values with 2^16-1
        interpolated_image[np.where(np.ma.getmask(mask_nodata))] = 2**16-1

        #Write to file
        outfile=options.output_interpolate.format(new_timestamp)
        date=new_timestamp[:8]
        time=new_timestamp[8:]
        write_interpolated_h5(outfile,interpolated_image,file_dict_interp,date,time)
        print 'Wrote', outfile


if __name__ == '__main__':

    #Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--first_precip_field',
                        help='Two-dimensional array containing the first precipitation field.')
    parser.add_argument('--second_precip_field',
                        help='Two-dimensional array containing the second precipitation field.')
    parser.add_argument('--seconds_between_steps',
                        type=int,
                        default=300,
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
                        default='test_{}.h5',
                        help='Output hdf5 file name, leave brackets {} for timestamp.')
    parser.add_argument('--output_sum',
                        default='test_5minsum_{}.h5',
                        help='Output sum hdf5 file name, leave brackets {} for timestamp.')


    options = parser.parse_args()
    main()
