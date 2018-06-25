import interpolation
import h5py
import hiisi
import numpy as np
import argparse
import datetime
import ConfigParser

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




def main():
    
    # FIRST LEARN HOW TO READ NETCDF OBS AND MODEL FIELDS IN. The function above is written only for reading HDF5 radar data.
    first_image_array, quantity, first_timestamp, mask_nodata = read_image("/fmi/data/nowcasting/testdata_radar/opera_rate/T_PAAH21_C_EUOC_20180613120000.hdf")
    second_image_array, quantity_second, second_timestamp, mask_nodata_second = read_image("/fmi/data/nowcasting/testdata_radar/opera_rate/T_PAAH21_C_EUOC_20180613121500.hdf")
    # DATA IS NOW LOADED AS A NORMAL NUMPY NDARRAY

    #Min and max values according to quantity (RATE/DBZH). These are defined in the argparse claim in the bottom of this program
    if quantity == 'RATE':
        quantity_min = options.R_min
        quantity_max = options.R_max
    if quantity == 'DBZH':
        quantity_min = options.DBZH_min
        quantity_max = options.DBZH_max
        print 'Quantity in input hdf5 file is DBZH. RATE needed.'
        exit(1)



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
