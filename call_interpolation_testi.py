import interpolate
import h5py
import hiisi
import numpy as np
import argparse
import datetime
import ConfigParser
import matplotlib.pyplot as plt
import netCDF4
import sys
import verif_calculation
import pandas as pd
import os

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
 

# def line_verif_plot(advdata,lindata,dmodata,perdata,parameter,verif_metrics_used):
#      # This plots the classic line plot of verification metrics. The plotted producers are
#      ## Interpolation done using AMV method (several sensitivity simulations are done)
#      ## Interpolation done using simple linear cross-dissolve
#      ## DMO
#      ## Persistence
#      # Plotting two plots:
#      # 1) Main producers for all different forecast lengths
#      # 2) Main producers and all sensitivity test values for a specific forecast length

#      # For testing purposes:

#      np.mean(fields_interpolated_advection[:,1,1,2,:,100:300,100:400],axis=(2,3))

#      advdata = verif_interpolated_advection
#      lindata = verif_interpolated_linear
#      dmodata = verif_interpolated_DMO
#      perdata = verif_interpolated_persistence

#      advdata = np.load("/fmi/data/nowcasting/results/verif_stats/verif_interpolated_advection_20180628060000_20180628120000_Temperature.npy")
#      lindata = np.load("/fmi/data/nowcasting/results/verif_stats/verif_interpolated_linear_20180628060000_20180628120000_Temperature.npy")
#      perdata = np.load("/fmi/data/nowcasting/results/verif_stats/verif_interpolated_persistence_20180628060000_20180628120000_Temperature.npy")
#      dmodata = np.load("/fmi/data/nowcasting/results/verif_stats/verif_interpolated_DMO_20180628060000_20180628120000_Temperature.npy")
#      advdata_default = advdata[:,1,1,2,:,:]
#      verif_metric = 'RMSE'

#      # 1) Main producers for all different forecast lengths
#      for verif_metric in verif_metrics_used:
#         verif_metric_index = int((verif_metrics_used==verif_metric).nonzero()[0])
#         df=pd.DataFrame({'AMV_CVdefault_2hours': advdata_default[0,:,verif_metric_index], 'AMV_CVdefault_3hours': advdata_default[1,:,verif_metric_index], 'AMV_CVdefault_4hours': advdata_default[2,:,verif_metric_index], 'AMV_CVdefault_5hours': advdata_default[3,:,verif_metric_index], 'AMV_CVdefault_6hours': advdata_default[4,:,verif_metric_index], 'linearCD_2hours': lindata[0,:,verif_metric_index], 'linearCD_3hours': lindata[1,:,verif_metric_index], 'linearCD_4hours': lindata[2,:,verif_metric_index], 'linearCD_5hours': lindata[3,:,verif_metric_index], 'linearCD_6hours': lindata[4,:,verif_metric_index], 'DMO': dmodata[:,verif_metric_index], 'persistence': perdata[:,verif_metric_index] })
# #        num_plots=df.shape[1]
       


#         # from matplotlib import cm
#         # cmap = cm.get_cmap('coolwarm_r')
#         # fig, ax = plt.subplots()
#         # plt.hold(True)
#         # df.iloc[:,0:4].plot(linewidth=2,cmap=cm.get_cmap('coolwarm_r'))
#         # # fig, ax = plt.subplots()
#         # ax.plot(df.iloc[:,3:10].plot(linewidth=2,cmap=cm.get_cmap('coolwarm_r'))
#         # # pp2 = df.iloc[:,3:10].plot(linewidth=2,cmap=cm.get_cmap('coolwarm_r'))
#         # pp = pd.concat([pp1, pp2], axis=1)
#         # plt.show()

#         # A plot without any refinement of colours etc.
#         from matplotlib import cm
#         df.plot()
        
#         # 1) Plot with all the main producers (plots presented as a function of predictability and hour) WHAT IS WRONG HERE!!!!???? ALL THE SCALES GET TOTALLY WRONG!!!!!
#         from matplotlib import cm
#         fig, ax = plt.subplots()
#         plt.hold(True)
#         df.iloc[:,[0,1,2,3,4]].plot(ax=ax,stacked=True,linewidth=2,colormap='GnBu')
#         # fig, ax = plt.subplots()
#         ax.plot(df.iloc[:,5],color='r',marker='o') #,label=list(df)[5])
#         df.iloc[:,[6,7,8,9,10]].plot(ax=ax,stacked=True,linewidth=2,colormap='YlOrRd')
#         ax.plot(df.iloc[:,11],color='b',marker='o') #,label=list(df)[5])
#         plt.show()
#         ax.legend(loc='best')
        
#         # 2) plots showing the sensitivity analysis to various AMV parameters
       






#     advdata_default=verif_interpolated_advection[:,1,1,2,:,:]
#     # Creating a pandas dataframe where the plotted data is being stored to
#     df=pd.DataFrame({'advdata_default_2': advdata_default[0,:,1], 'advdata_default_3': advdata_default[1,:,1], 'advdata_default_4': advdata_default[2,:,1], 'advdata_default_5': advdata_default[3,:,1], 'advdata_default_5': advdata_default[4,:,1] })

#     plt.plot(df)
#     plt.legend()


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
    
    # #Read parameters from config file for interpolation (or optical flow algorithm, find out this later!). The function for reading the parameters is defined above.
    farneback_params=farneback_params_config()

    # FIRST LEARN HOW TO READ NETCDF OBS AND MODEL FIELDS IN. The function below is written only for reading HDF5 radar data. Also radar observations need to be in some kind of accumulation form as pal_skandinavia data are always 1h accumulations.

    # In the "verification mode", the idea is to load in the "observational" and "forecast" datasets as numpy arrays. Both of these numpy arrays ("image_array") contain ALL the timesteps contained also in the files themselves. In addition, the returned variables "timestamp" and "mask_nodata" contain the values for all the timesteps.

    # First precipitation field is from Tuliset2/analysis. What about possible scaling???
    if options.parameter == 'Precipitation1h_TULISET': # This needs to be done better
        image_array1, quantity1_min, quantity1_max, timestamp1, mask_nodata1 = read_HDF5("/fmi/data/nowcasting/testdata_radar/opera_rate/T_PAAH21_C_EUOC_20180613120000.hdf")
        # image_array2, quantity2_min, quantity2_max, timestamp2, mask_nodata2 = read_HDF5("/fmi/data/nowcasting/testdata_radar/opera_rate/T_PAAH21_C_EUOC_20180613121500.hdf")
    else:
        image_array1, quantity1_min, quantity1_max, timestamp1, mask_nodata1, nodata1 = read_nc(options.obsdata)
        quantity1 = options.parameter
    # The second field is always a model forecast field
    image_array2, quantity2_min, quantity2_max, timestamp2, mask_nodata2, nodata2 = read_nc(options.modeldata)
    quantity2 = options.parameter

    # In verification mode timestamps must be the same, otherwise exiting!
    if sum(timestamp1 == timestamp2)!=timestamp1.shape[0]:
       raise ValueError("obs and model data have different timestamps!")
       # sys.exit( "Timestamps do not match!" )    

    # Missing_values of image_array2 ("nodata2") are changed to "nodata1". The larger data is made smaller so that the actual data points in the two data sets have the same geographical domain (LAPS data covers a slighlty larger domain than pal).
    if np.ma.count_masked(mask_nodata2) > np.ma.count_masked(mask_nodata1):
        image_array1[np.where( np.ma.getmask(mask_nodata2) )] = nodata1
        image_array2[np.where( np.ma.getmask(mask_nodata2) )] = nodata1
        mask_nodata1 = mask_nodata2
    else:
        image_array1[np.where( np.ma.getmask(mask_nodata1) )] = nodata1
        image_array2[np.where( np.ma.getmask(mask_nodata1) )] = nodata1
        mask_nodata2 = mask_nodata1
    # Missing data values are the same for the two fields, and it is taken from the first field.
    nodata = nodata2 = nodata1
    
    # Defining definite min/max values from the two fields
    R_min=min(quantity1_min,quantity2_min)
    R_max=max(quantity1_max,quantity2_max)
    # Defining a masked array that is the same for all the forecast fields and both producers (if even one forecast length is missing for a specific grid point, that grid point is masked out).
    mask_nodata = np.sum(np.ma.getmask(mask_nodata1),axis=0)>0
    mask_nodata = np.ma.masked_where(mask_nodata == True,mask_nodata)
    
    # Making another arbitrary mask for verification purposes (avoiding possible boundary problems)
    mask_nodata_middle=np.ones(mask_nodata.shape,dtype=bool)
    mask_nodata_middle[100:300,100:400]=False
    mask_nodata_middle = np.ma.masked_where(mask_nodata_middle == True,mask_nodata_middle)

    # # Mask nodata values to be just slightly less than the lowest field value
    # image_array1[np.where(np.ma.getmask(mask_nodata1))] = R_min
    # nodata = -nodata

    # DATA IS NOW LOADED AS NORMAL NUMPY NDARRAYS







    # RANGE OF POSSIBLE VALUES FOR SENSITIVITY TESTS
    # With this setup the sensitivity analysis is done by running AMV calculation with 200 different set-ups. The result array of ones is about 2218 MBs, so it still should be easily manageable.
    predictabilitys = np.arange(2,options.predictability+1) # # There's no point in doing calculations for less than 2 hours: One hour predictability on a one hour time resolution would just mean that the first timestep is the analysis and the second one the model forecast.
    fb_winsizes = np.array([10,30,50,70,100]) # Averaging window size
    fb_levelss = np.array([1,6]) # Number of pyramid layers including the initial image
    fb_poly_ns = np.array([3,5,7,20]) # Size of the pixel neighborhood used to find polynomial expansion in each pixel. CAUTIONARY AT FIRST! np.array([3,5,7,20]) # np.array([7])

    # USED VERIFICATION METRICS
    verif_metrics_used = np.array(["ME","RMSE"])

    # RESULT ARRAYS
    # Interpolated fields and verification arrays for the AMV method, including sensitivity analysis
    fields_interpolated_advection = (np.ones(tuple([len(predictabilitys),len(fb_winsizes),len(fb_levelss),len(fb_poly_ns)])+image_array1.shape))
    verif_interpolated_advection = (np.ones(tuple([len(predictabilitys),len(fb_winsizes),len(fb_levelss),len(fb_poly_ns),image_array1.shape[0],verif_metrics_used.shape[0]])))

    # Interpolated fields and verification arrays for the linear cross-dissolve method
    fields_interpolated_linear = (np.ones(tuple([len(predictabilitys)])+image_array1.shape))
    verif_interpolated_linear = (np.ones(tuple([len(predictabilitys),image_array1.shape[0],verif_metrics_used.shape[0]]))) 
    # Verification array for the DMO forecast
    verif_interpolated_DMO = (np.ones(tuple([image_array1.shape[0],verif_metrics_used.shape[0]]))) 
    # Verification array for the persistence forecast
    verif_interpolated_persistence = (np.ones(tuple([image_array1.shape[0],verif_metrics_used.shape[0]]))) 
    


    # CALCULATING INTERPOLATED IMAGES FOR DIFFERENT PRODUCERS AND CALCULATING VERIF METRICS
    for predictability in predictabilitys:
       for fb_winsize in fb_winsizes:
          for fb_levels in fb_levelss:
             for fb_poly_n in fb_poly_ns:
                # Like mentioned at https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html, a reasonable value for poly_sigma depends on poly_n. Here we use a fraction 4.6 for these values.
                fb_poly_sigma = fb_poly_n / 4.6
                fb_params = (farneback_params[0],fb_levels,fb_winsize,farneback_params[3],fb_poly_n,fb_poly_sigma,farneback_params[6])
                # Interpolated data
                interpolated_advection=interpolate.advection(obsfields=image_array1, modelfields=image_array2, mask_nodata=mask_nodata, farneback_params=fb_params, predictability=predictability, seconds_between_steps=options.seconds_between_steps, R_min=R_min, R_max=R_max, missingval=nodata, logtrans=False)
                fields_interpolated_advection[((predictabilitys==predictability).nonzero()),((fb_winsizes==fb_winsize).nonzero()),((fb_levelss==fb_levels).nonzero()),((fb_poly_ns==fb_poly_n).nonzero()),:,:,:] = interpolated_advection
                # verif metrics calculated from interpolated data DONE BY HAND HERE!!!!
                verif_interpolated_advection[((predictabilitys==predictability).nonzero()),((fb_winsizes==fb_winsize).nonzero()),((fb_levelss==fb_levels).nonzero()),((fb_poly_ns==fb_poly_n).nonzero()),:,0] = verif_calculation.ME(image_array1,interpolated_advection,mask_nodata_middle)
                verif_interpolated_advection[((predictabilitys==predictability).nonzero()),((fb_winsizes==fb_winsize).nonzero()),((fb_levelss==fb_levels).nonzero()),((fb_poly_ns==fb_poly_n).nonzero()),:,1] = verif_calculation.RMSE(image_array1,interpolated_advection,mask_nodata_middle)
                print(predictability,fb_winsize,fb_levels,fb_poly_n)
       # Calculating blend through simple linear cross-dissolve
       # Interpolated data
       interpolated_linear=interpolate.linear(obsfields=image_array1, modelfields=image_array2, mask_nodata=mask_nodata, predictability=predictability, seconds_between_steps=options.seconds_between_steps, missingval=nodata)
       fields_interpolated_linear[((predictabilitys==predictability).nonzero()),:,:,:] = interpolated_linear
       # verif metrics calculated from interpolated data DONE BY HAND HERE!!!!
       verif_interpolated_linear[((predictabilitys==predictability).nonzero()),:,0] = verif_calculation.ME(image_array1,interpolated_linear,mask_nodata_middle)
       verif_interpolated_linear[((predictabilitys==predictability).nonzero()),:,1] = verif_calculation.RMSE(image_array1,interpolated_linear,mask_nodata_middle)
    # Calculate verification metrics for DMO forecasts
    verif_interpolated_DMO[:,0] = verif_calculation.ME(image_array1,image_array2,mask_nodata_middle)
    verif_interpolated_DMO[:,1] = verif_calculation.RMSE(image_array1,image_array2,mask_nodata_middle)
    # Calculate verification metrics for persistence forecast
    verif_interpolated_persistence[:,0] = verif_calculation.ME(image_array1,np.tile(image_array1[0,:,:],(image_array1.shape[0],1,1)),mask_nodata_middle)
    verif_interpolated_persistence[:,1] = verif_calculation.RMSE(image_array1,np.tile(image_array1[0,:,:],(image_array1.shape[0],1,1)),mask_nodata_middle)
    

    # laske_eri_verif_metriikat voisi laskea metriikan myos liikevektorikenttien perusteella? Ensin kuitenkin perus-RMSE:n kaltaiset laskentaan...taman jalkeen muuta. Jos tekee, pitaa muokata interpolate.py -skriptia palauttamaan liikevektorikentat, jotta voi laskea myos analyysikenttien ja blendikenttien valille liikevektorit.
                

    # # For error hunting, calculate and plot a data frame which shows the mean values of the fields for various time steps
    # advdata_default = np.mean(fields_interpolated_advection[:,1,1,2,:,100:300,100:400],axis=(2,3))
    # lindata = np.mean(fields_interpolated_linear[:,:,100:300,100:400],axis=(2,3))
    # dmodata = np.mean(image_array2[:,100:300,100:400],axis=(1,2))
    # perdata = np.tile(image_array1[0,:,:],(image_array1.shape[0],1,1))
    # perdata = np.mean(perdata[:,100:300,100:400],axis=(1,2))

    # advdata_default = (fields_interpolated_advection[:,1,1,2,:,200,200])
    # lindata = (fields_interpolated_linear[:,:,200,200])
    # dmodata = (image_array2[:,200,200])
    # perdata = np.tile(image_array1[0,:,:],(image_array1.shape[0],1,1))
    # perdata = (perdata[:,200,200])
    # verif_metric = 'RMSE'
    # df=pd.DataFrame({'AMV_CVdefault_2hours': advdata_default[0,:], 'AMV_CVdefault_3hours': advdata_default[1,:], 'AMV_CVdefault_4hours': advdata_default[2,:], 'AMV_CVdefault_5hours': advdata_default[3,:], 'AMV_CVdefault_6hours': advdata_default[4,:], 'linearCD_2hours': lindata[0,:], 'linearCD_3hours': lindata[1,:], 'linearCD_4hours': lindata[2,:], 'linearCD_5hours': lindata[3,:], 'linearCD_6hours': lindata[4,:], 'DMO': dmodata, 'persistence': perdata })
    # df.plot()
    # df.iloc[:,[4,10,5]].plot()


    
    # # This is only for a quick visual inspection of the plots
    # # TEMPERATURE
    # # contour plots with 3-hour predictability and with severl forecast lengths
    # plt.subplot(2,2,1)
    # plt.imshow(fields_interpolated_advection[1,1,1,2,0,:,:],vmin=270,vmax=310,cmap='hot')
    # plt.subplot(2,2,2)
    # plt.imshow(fields_interpolated_advection[1,1,1,2,1,:,:],vmin=270,vmax=310,cmap='hot')
    # plt.subplot(2,2,3)
    # plt.imshow(fields_interpolated_advection[1,1,1,2,2,:,:],vmin=270,vmax=310,cmap='hot')
    # plt.subplot(2,2,4)
    # plt.imshow(fields_interpolated_advection[1,1,1,2,3,:,:],vmin=270,vmax=310,cmap='hot')
    # plt.show()
    
    # # mean RMSE of all AMV calculated methods for various predictabilities and forecast lengths
    # print(np.mean(verif_interpolated_advection[:,:,:,:,:,1],axis=(1,2,3)))
    # # mean RMSE of AMV method using default set-up parameters for various predictabilities and forecast lengths
    # print(verif_interpolated_advection[:,1,1,2,:,1])
    # # mean RMSE of linear cross-dissolve method for various predictabilities and forecast lengths
    # print(verif_interpolated_linear[:,:,1])
    # # mean RMSE of DMO for various forecast lengths
    # print(verif_interpolated_DMO[:,1])
    # # mean RMSE of persistence for various forecast lengths
    # print(verif_interpolated_persistence[:,1])


    # # PRESSURE
    # plt.subplot(2,2,1)
    # plt.imshow(fields_interpolated_advection[4,1,0,0,0,:,:],vmin=100500,vmax=101900,cmap='hot')
    # plt.subplot(2,2,2)
    # plt.imshow(fields_interpolated_advection[4,1,0,0,1,:,:],vmin=100500,vmax=101900,cmap='hot')
    # plt.subplot(2,2,3)
    # plt.imshow(fields_interpolated_advection[4,1,0,0,2,:,:],vmin=100500,vmax=101900,cmap='hot')
    # plt.subplot(2,2,4)
    # plt.imshow(fields_interpolated_advection[4,1,0,0,3,:,:],vmin=100500,vmax=101900,cmap='hot')
    # plt.show()

    # # Saving of the verif results to files. Also display of a "classic line plot"
    # line_verif_plot(verif_interpolated_advection,verif_interpolated_linear,verif_interpolated_dmo,verif_interpolated_persistence,optionsparameter,verif_metrics_used)

    # Saving the verif stats to a file, scp'ing them to devmos.fmi.fi and removing the local result files
    remotehost="ylhaisi@devmos.fmi.fi"
    remotedir="/data/statcal/results/nowcasting/verif_stats/"
    outdir="/fmi/data/nowcasting/results/verif_stats/"

    outfile = "verif_interpolated_advection_" + timestamp1[0].strftime("%Y%m%d%H%M%S") + "_" + timestamp1[timestamp1.shape[0]-1].strftime("%Y%m%d%H%M%S") + "_" + options.parameter + ".npy"
    np.save(outdir + outfile,verif_interpolated_advection)
    os.system('scp "%s" "%s:%s"' % (outdir + outfile, remotehost, remotedir + outfile) )
    os.system('rm "%s"' % (outdir + outfile) )
    outfile = "verif_interpolated_linear_" + timestamp1[0].strftime("%Y%m%d%H%M%S") + "_" + timestamp1[timestamp1.shape[0]-1].strftime("%Y%m%d%H%M%S") + "_" + options.parameter + ".npy"
    np.save(outdir + outfile,verif_interpolated_linear)
    os.system('scp "%s" "%s:%s"' % (outdir + outfile, remotehost, remotedir + outfile) )
    os.system('rm "%s"' % (outdir + outfile) )
    outfile = "verif_interpolated_DMO_" + timestamp1[0].strftime("%Y%m%d%H%M%S") + "_" + timestamp1[timestamp1.shape[0]-1].strftime("%Y%m%d%H%M%S") + "_" + options.parameter + ".npy"
    np.save(outdir + outfile,verif_interpolated_DMO)
    os.system('scp "%s" "%s:%s"' % (outdir + outfile, remotehost, remotedir + outfile) )
    os.system('rm "%s"' % (outdir + outfile) )
    outfile = "verif_interpolated_persistence_" + timestamp1[0].strftime("%Y%m%d%H%M%S") + "_" + timestamp1[timestamp1.shape[0]-1].strftime("%Y%m%d%H%M%S") + "_" + options.parameter + ".npy"
    np.save(outdir + outfile,verif_interpolated_persistence)
    os.system('scp "%s" "%s:%s"' % (outdir + outfile, remotehost, remotedir + outfile) )
    os.system('rm "%s"' % (outdir + outfile) )
    
    

    # ssh ylhaisi@devmos.fmi.fi "touch /data/statcal/results/nowcasting/testi.bar"






    # #Calculate number of interpolated frames from timestamps and minutes between steps
    # formatted_time_first=datetime.datetime.strptime(timestamp1,'%Y%m%d%H%M%S')
    # formatted_time_second=datetime.datetime.strptime(timestamp2,'%Y%m%d%H%M%S')
    # timediff_seconds=(formatted_time_second-formatted_time_first).total_seconds()
    # n_interp_frames=int(timediff_seconds/options.seconds_between_steps-1) # Here, int type (timediff_seconds) is divided with another int 
    # type (seconds_between_steps) so the result is int as well

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
                        default="/fmi/data/nowcasting/obsdata.nc",
                        help='Obs data, representing the first time step used in image morphing.')
    parser.add_argument('--modeldata',
                        default="/fmi/data/nowcasting/modeldata.nc",
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
                        type=int,
                        default='6',
                        help='Predictability in hours. Between the analysis and forecast of this length, forecasts need to be interpolated')
    parser.add_argument('--parameter',
                        default='Temperature',
                        help='Variable which is handled.')
    parser.add_argument('--mode',
                        default='verif',
                        help='Either "verif" or "fcst" mode. In verification mode, verification statistics are calculated from the blended forecasts. In forecast mode no.')


    options = parser.parse_args()
    main()
