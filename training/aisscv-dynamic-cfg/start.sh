#!/bin/bash
CURRENT_PATH="/"
COLOR=0
BATCH_SIZE=3000
FOLDS=1
NAME=$(date +%Y%m%d_%H%M%S)
NBR_AUGMENTATIONS=10
DARKNET="/darknet"

echo $@
for arg in "$@"
do
    case $arg in
        -path=*|--path=*)
        CURRENT_PATH="${arg#*=}"
        shift # Remove argument value from processing
        ;;
        -darknet=*|--darknet_path=*)
        DARKNET="${arg#*=}"
        shift # Remove argument value from processing
        ;;
        -c|--color)
        COLOR=1
        shift # Remove --initialize from processing
        ;;
        -b=*|--batch_size=*)
        BATCH_SIZE="${arg#*=}"
        shift # Remove argument value from processing
        ;;
        -f=*|--folds=*)
        FOLDS="${arg#*=}"
        shift # Remove argument name from processing
        ;;
        -n=*|--name=*)      
        NAME="${arg#*=}"
        shift # Remove argument name from processing
        ;;
        -a=*|--augmentations=*)
        NBR_AUGMENTATIONS="${arg#*=}"
        shift # Remove argument name from processing
        ;;
    esac
done
(cd "$CURRENT_PATH" && git clone https://github.com/pineappledafruitdude/AISSCV.git aisscv)
(cd "$CURRENT_PATH"aisscv/preprocessing/ &&
python3.8 run_train.py -n="$NAME" -o "/aisscv/model" -c="$COLOR" -f="$FOLDS" -nbr_augment="$NBR_AUGMENTATIONS" -darknet "$DARKNET")