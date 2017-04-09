#!/bin/bash

if which nvidia-docker > /dev/null
then
    true
else
    echo nvidia-docker not installed
    echo https://github.com/NVIDIA/nvidia-docker
    exit
fi

if nvidia-docker inspect aaalgo/plumo >& /dev/null
then
    true
else
    echo "Downloading aaalgo/plumo docker image, please be patient...."
    nvidia-docker pull aaalgo/plumo
fi

INPUT=$1
OUTPUT=$2

if [ -z "$OUTPUT" ]
then
    echo Usage:
    echo "    process.sh  INPUT OUTPUT"
    echo
    echo "INPUT: directory contain dcm files"
    echo "OUTPUT: output directory"
    exit
fi

if [ ! -d "$INPUT" ]
then
    echo Input directory $INPUT does not exist.
    exit
fi


#if [ -e "$OUTPUT" ]
#then
#    echo Output directory $OUTPUT exists, not overwriting
#    exit
#fi


INPUT=`readlink -e $INPUT`

if echo "$OUTPUT" | grep -v '^/'
then
    OUTPUT=$PWD/$OUTPUT
fi

MY_UID=`id -u`
MY_GID=`id -g`

mkdir -p $OUTPUT
nvidia-docker run -u=$MY_UID:$MY_GID -v $INPUT:/input -v $OUTPUT:/output aaalgo/plumo /adsb3/run.sh


