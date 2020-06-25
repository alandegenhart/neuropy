#!/bin/sh

# Define data paths
SOURCE_DIR="/afs/ece.cmu.edu/project/nspg/adegenha/results/el_ms/fig_4/flow_10D"
DEST_DIR="/Users/alandegenhart/results/el_ms/fig_4/"

# Secure copy
scp -r -P 22 adegenha@axon.ece.cmu.edu:$SOURCE_DIR $DEST_DIR
