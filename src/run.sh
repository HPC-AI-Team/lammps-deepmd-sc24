#!/bin/bash


module sw lang/tcsds-1.2.38

./compile.sh

cd BIN_deepmd_water

pjsub ./lmp_deepmd_pjsub

cd ../BIN_deepmd_copper

pjsub ./lmp_deepmd_pjsub
