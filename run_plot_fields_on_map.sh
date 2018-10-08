#!/bin/bash

#Paths
PYTHON=${PYTHON:-'/fmi/dev/python_virtualenvs/venv/bin/python'}

#Variables
DATAPATH=${DATAPATH:-"$PWD"}
OUTPATH=${OUTPATH:-"$PWD"}
USED_OBS=${USED_OBS:-"laps"}
USED_MODEL=${USED_MODEL:-"pal"}
DOMAIN=${DOMAIN:-"SCAND2"}

INPUT_FILE="201808240700_endtime201808241300_laps_skandinavia_DOMAIN=SCAND2_Temperature.nc"
OUTPUT_FILE="laps"
echo $DATAPATH/$INPUT_FILE
cmd="$PYTHON plot_fields_on_map.py --input_file $DATAPATH/$INPUT_FILE --output_file $OUTPATH/$OUTPUT_FILE"
echo $cmd
eval $cmd

#INPUT_FILE="201808240700_fcst201808241300_pal_skandinavia_DOMAIN=SCAND2_Temperature.nc"
#OUTPUT_FILE="pal"
#echo $DATAPATH/$INPUT_FILE
#cmd="$PYTHON plot_fields_on_map.py --input_file $DATAPATH/$INPUT_FILE --output_file $OUTPATH/$OUTPUT_FILE"
#echo $cmd
#eval $cmd
