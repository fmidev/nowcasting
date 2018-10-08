#!/bin/bash

#Paths                                                                                                                                      
PYTHON=${PYTHON:-'/fmi/dev/python_virtualenvs/venv/bin/python'}

#Variables 
DATAPATH=${DATAPATH:-"$PWD"}
OUTPATH=${OUTPATH:-"$PWD"}
ADV_FILE="verif_interpolated_advection_20180907030000_20180907090000_Pressure.npy"
INPUT_FILE=$ADV_FILE
OUTPUT_FILE="test_verif_adv.png"

#Call plotting
echo $DATAPATH/$INPUT_FILE
cmd="$PYTHON plot_verification_scores.py --input_file $DATAPATH/$INPUT_FILE --output_file $OUTPATH/$OUTPUT_FILE"
echo $cmd
eval $cmd
