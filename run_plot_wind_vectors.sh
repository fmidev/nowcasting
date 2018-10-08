#!/bin/bash

#Paths
PYTHON=${PYTHON:-'/fmi/dev/python_virtualenvs/venv/bin/python'}

#Variables
DATAPATH=${DATAPATH:-"$PWD"}
OUTPATH=${OUTPATH:-"$PWD"}
INPUT_FILE_WIND_U="201809180500_endtime201809181100_laps_skandinavia_DOMAIN=SCAND2_WindUMS.nc"
INPUT_FILE_WIND_V="201809180500_endtime201809181100_laps_skandinavia_DOMAIN=SCAND2_WindVMS.nc"
OUTPUT_FILE="test_winds.png"

echo $DATAPATH/$INPUT_FILE_WIND_U
echo $DATAPATH/$INPUT_FILE_WIND_V

cmd="$PYTHON plot_wind_vectors.py --input_file_u $DATAPATH/$INPUT_FILE_WIND_U --input_file_v $DATAPATH/$INPUT_FILE_WIND_V --output_file $OUTPATH/$OUTPUT_FILE"
echo $cmd
eval $cmd
