#!/bin/bash
#Calculate interpolated fields between two radar composites.
#Tuuli.Perttula@fmi.fi

#Define paths
PYTHON=${PYTHON:-'/fmi/dev/python_virtualenvs/venv/bin/python'}
DATAPATH=${DATAPATH:-'/radar/storage/HDF5/tuliset2dev/testdata/20120925'}

#Input data
IMAGE1=${IMAGE1:-'T_PAAH21_C_EUOC_20120925053000.hdf'}
IMAGE2=${IMAGE2:-'T_PAAH21_C_EUOC_20120925054500.hdf'}
SECONDS_BETWEEN_STEPS=${SECONDS_BETWEEN_STEPS:-30}

#Output
OUTPATH=${OUTPATH:-$PWD}
OUTFILE_INTERP=${OUTFILE_INTERP:-'{}_DOMAIN=ODC.h5'}
OUTFILE_SUM=${OUTFILE_SUM:-'{}_5minsum_DOMAIN=ODC.h5'}

#Call interpolation
cmd="$PYTHON call_interpolation.py --first_precip_field $DATAPATH/$IMAGE1 --second_precip_field $DATAPATH/$IMAGE2 --seconds_between_steps $SECONDS_BETWEEN_STEPS --output_interpolate $OUTPATH/$OUTFILE_INTERP --output_sum $OUTPATH/$OUTFILE_SUM"
echo $cmd
eval $cmd

#Convert output to png
OUTFILES=`echo $OUTFILE | awk -F{} '{print $2}'`

#for f in $OUTPATH/*$OUTFILES;do
#    PNGFILE=`basename $f .h5`.png
#    rack $f --encoding C --convert --iResize 496,731 -o $OUTPATH/$PNGFILE
#done
