#!/bin/bash

#Paths
PYTHON=${PYTHON:-'/fmi/dev/python_virtualenvs/venv/bin/python'}

#Variables
DATAPATH=${DATAPATH:-"$PWD"}
OUTPATH=${OUTPATH:-"$PWD"}
#INPUT_FILE="201808170400_endtime201808171000_laps_skandinavia_DOMAIN=SCAND2_Temperature.nc"
INPUT_FILE="201808240000_endtime201808240600_laps_skandinavia_DOMAIN=SCAND2_Temperature.nc"

OUTPUT_FILE="test.png"

#Run plotting
echo $DATAPATH/$INPUT_FILE
cmd="$PYTHON plot_fields_on_map.py --input_file $DATAPATH/$INPUT_FILE --output_file $OUTPATH/$OUTPUT_FILE"
echo $cmd
eval $cmd
