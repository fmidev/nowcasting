# -*- coding: utf-8 -*- 
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import netCDF4
import argparse
import numpy as np


def plot_verif_scores(fc_lengths,verif_scores,labels,outfile,title):

    for n in range(0, verif_scores.shape[0]): 
        plt.plot(fc_lengths, verif_scores[n,:], linewidth=2.0, label=str(labels[n]))

    plt.legend(bbox_to_anchor=(0.21, 1))
    plt.xlabel('Forecast length (h)')
    plt.ylabel('Units')
    plt.title(title)
    plt.savefig(outfile,bbox_inches='tight', pad_inches=0)
    plt.close()


def main():

    verif_scores_all=np.load(options.input_file)
 
    fc_lengths=np.arange(0,6+1)

    rmse_varying_predictability=verif_scores_all[:,1,1,2,:,1]
    rmse_varying_winsize=verif_scores_all[3,:,1,2,:,1]
    rmse_varying_levels=verif_scores_all[3,1,:,2,:,1]
    rmse_varying_poly_ns=verif_scores_all[3,1,1,:,:,1]

    me_varying_predictability=verif_scores_all[:,1,1,2,:,0]
    me_varying_winsize=verif_scores_all[3,:,1,2,:,0]
    me_varying_levels=verif_scores_all[3,1,:,2,:,0]
    me_varying_poly_ns=verif_scores_all[3,1,1,:,:,0]

    #(5, 5, 2, 4, 7, 2)
    #Predictability, fb_winsize, fb_levels, fb_poly_ns, fc length, used verif metric

    predictabilities = np.arange(2,2+5)
    fb_winsizes = np.array([10,30,50,70,100]) # Averaging window size, default: 30
    fb_levelss = np.array([1,6]) # Number of pyramid layers including the initial image, default: 6
    fb_poly_ns = np.array([3,5,7,20]) # Size of the pixel neighborhood used to find polynomial expansion in each pixel. default: 7

    #verif_metrics = np.array(["ME","RMSE"])

    plot_verif_scores(fc_lengths,rmse_varying_predictability,predictabilities,'rmse_advection_predictability_20180912010000_20180912070000_Pressure.png','RMSEPressure, Varying predictability')
    plot_verif_scores(fc_lengths,rmse_varying_winsize,fb_winsizes,'rmse_advection_winsize_20180912010000_20180912070000_Pressure.png','RMSE\
Pressure, RMSE Pressure, Varying Farneback winsize')
    plot_verif_scores(fc_lengths,rmse_varying_levels,fb_levelss,'rmse_advection_levels_20180912010000_20180912070000_Pressure.png','RMSE Pressure, Varying Farneback levels')
    plot_verif_scores(fc_lengths,rmse_varying_poly_ns,fb_poly_ns,'rmse_advection_polyns_20180912010000_20180912070000_Pressure.png','RMSE Pressure, Varying Farneback Poly ns')

    plot_verif_scores(fc_lengths,me_varying_predictability,predictabilities,'me_advection_predictability_20180912010000_20180912070000_Pressure.png','ME Pressure, Varying predictability')
    plot_verif_scores(fc_lengths,me_varying_winsize,fb_winsizes,'me_advection_winsize_20180912010000_20180912070000_Pressure.png','ME Pressure, Varying Farneback winsize')
    plot_verif_scores(fc_lengths,me_varying_levels,fb_levelss,'me_advection_levels_20180912010000_20180912070000_Pressure.png','ME Pressure, Varying Farneback levels')
    plot_verif_scores(fc_lengths,me_varying_poly_ns,fb_poly_ns,'me_advection_polyns_20180912010000_20180912070000_Pressure.png','ME Pressure, Varying Farneback Poly ns')



if __name__ == '__main__':

    #Parse commandline arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        help='Input file to plot.')
    parser.add_argument('--output_file',
                        help='Output png file.')

    options = parser.parse_args()
    main()
