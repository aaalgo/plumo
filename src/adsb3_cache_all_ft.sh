#!/bin/bash

# This program generates feature vector of various
# combination of NODULE model and FEATURE model.
# The generated feature files are cached in the feature
# directory.
# only the feature data actually used by
# predict1 and predict2 have to be generated.


# Feature directory naming scheme
# comp1_comp2_[m?]_b8[_hull][_old]
# comp1 is the nodule model name
# comp2 is the feature model name
# both nodule and feature models are in models directory,
# if the nodule is detected under sagittal or coronal view,
# m2 or m3 is added.  If the nodule detection is applied
# with a segmentation mask/hull, _mask or _hull is added.
# so the feature directory name fully determines how
# to invoke adsb3_cache_ft.py.

# _old means to use adsb3_cache_ft_old.py instead of adsb3_cache_ft.py for extraction.

# _b8 and --bits 8: we only use 8-bit image here.
# --fts_dropout.  for feature model with dropout this have to
# be specified.  A model either works with --fts_dropout, or not,
# but not both.

# the exit command in this file are for partial processing,
# must be modified according to actual usage of feature data.

./adsb3_cache_ft.py --bits 8 --prob unet_k 
./adsb3_cache_ft.py --bits 8 --prob unet --mode 2
./adsb3_cache_ft.py --bits 8 --prob unet --mode 3
./adsb3_cache_ft.py --bits 8 --prob tiny --fts ft1 --fts_dropout
./adsb3_cache_ft.py --bits 8 --prob tiny --mode 4

# the above are needed to reproduce kaggle submission

exit

if false
then
./adsb3_cache_ft.py --bits 8 --prob tiny --fts ft1 --fts_dropout
./adsb3_cache_ft.py --bits 8 --prob tiny --mode 2
./adsb3_cache_ft.py --bits 8 --prob tiny --mode 3
./adsb3_cache_ft.py --bits 8 --prob unet_k --fts ft1 --fts_dropout
./adsb3_cache_ft.py --bits 8 --prob tiny --mode 4
./adsb3_cache_ft.py --bits 8 --prob tiny --fts ft --fts_dropout
fi


./adsb3_cache_ft.py --bits 8 --prob tiny --mask hull
exit
./adsb3_cache_ft.py --bits 8 --prob tiny
./adsb3_cache_ft.py --bits 8 --prob tiny --mode 2
./adsb3_cache_ft.py --bits 8 --prob tiny --mode 3
./adsb3_cache_ft.py --bits 8 --prob tiny --mode 3 --fts ft --fts_dropout

exit
./adsb3_cache_ft.py --bits 8 --prob unet_k --mask hull
./adsb3_cache_ft.py --bits 8 --prob unet --fts ft --fts_dropout --mask hull
./adsb3_cache_ft.py --bits 8 --prob unet --mode 4 --mask hull
./adsb3_cache_ft.py --bits 8 --prob unet --mode 3 --mask hull
exit

#./adsb3_cache_ft.py --bits 8 --prob unet  --mask hull
./adsb3_cache_ft.py --bits 8 --prob unet_k --fts ft --fts_dropout --mask hull
./adsb3_cache_ft.py --bits 8 --prob unet --mode 2 --fts ft --fts_dropout --mask hull
./adsb3_cache_ft.py --bits 8 --prob unet --mode 2 --mask hull
exit

./adsb3_cache_ft.py --bits 8 --prob unet_k --fts ft --fts_dropout
./adsb3_cache_ft.py --bits 8 --prob unet_k
./adsb3_cache_ft.py --bits 8 --prob unet --mode 2 --fts ft --fts_dropout
./adsb3_cache_ft.py --bits 8 --prob unet --mode 2
./adsb3_cache_ft.py --bits 8 --prob unet 
./adsb3_cache_ft.py --bits 8 --prob unet --fts ft --fts_dropout
./adsb3_cache_ft.py --bits 8 --prob unet --mode 4
./adsb3_cache_ft.py --bits 8 --prob unet --mode 3

exit

exit
./adsb3_cache_ft.py --bits 8 --prob small  --mode 4 --fts smallft
./adsb3_cache_ft.py --bits 8 --prob small  --mode 3 --fts smallft
./adsb3_cache_ft.py --bits 8 --prob small  --mode 2 --fts smallft
exit

./adsb3_cache_ft.py --bits 8 --prob small  --mode 3
./adsb3_cache_ft.py --bits 8 --prob small  --mode 2
./adsb3_cache_ft.py --bits 8 --prob small  --mode 4
#./adsb3_cache_ft.py --bits 8 --prob small  --mode 1
exit
./adsb3_cache_ft.py --bits 8 --prob small  --mode 1 --fts ft2 --fts_dropout
exit
./adsb3_cache_ft.py --bits 8 --prob small_4k  --mode 2
exit
./adsb3_cache_ft.py --bits 8 --prob nnc  --mode 1
exit
./adsb3_cache_ft.py --bits 8 --prob small_4k  --mode 3
exit
./adsb3_cache_ft.py --bits 8 --prob small_4k  --mode 1
exit
./adsb3_cache_ft.py --bits 8 --prob small  --mode 1 --fts ft --fts_dropout
exit
./adsb3_cache_ft.py --bits 8 --prob small  --mode 1
exit
./adsb3_cache_ft.py --bits 8 --prob tiny2  --mode 1 --fts tcia --fts_dropout
exit
./adsb3_cache_ft.py --bits 8 --prob tiny2  --mode 1
exit
./adsb3_cache_ft.py --bits 8 --prob unet5 --fts tcia --fts_dropout
exit
./adsb3_cache_ft_calibc.py --bits 8 --prob unet --mode 3
./adsb3_cache_ft_calibc.py --bits 8 --prob unet --mode 1 --fts ft --fts_dropout
./adsb3_cache_ft_calibc.py --bits 8 --prob unet --mode 2 --fts ft --fts_dropout
./adsb3_cache_ft_calibc.py --bits 8 --prob unet --mode 3 --fts ft --fts_dropout
./adsb3_cache_ft_calibc.py --bits 8 --prob unet --mode 4 --fts ft --fts_dropout
./adsb3_cache_ft_calibc.py --bits 8 --prob unet --mode 1
./adsb3_cache_ft_calibc.py --bits 8 --prob unet --mode 2
./adsb3_cache_ft_calibc.py --bits 8 --prob unet --mode 4
./adsb3_cache_ft_calibc.py --bits 8 --prob unet --mode 1 --fts ft_k --fts_dropout
./adsb3_cache_ft_calibc.py --bits 8 --prob unet --mode 2 --fts ft_k --fts_dropout
./adsb3_cache_ft_calibc.py --bits 8 --prob unet --mode 3 --fts ft_k --fts_dropout
./adsb3_cache_ft_calibc.py --bits 8 --prob unet --mode 4 --fts ft_k --fts_dropout
exit
./adsb3_cache_ft_calibc.py --bits 8 --prob unet_l #--mask hull
exit
./adsb3_cache_ft_calibc.py --bits 8 --prob unet_k #--mask hull
exit

./adsb3_cache_ft_batch.py --bits 8 --fts ft_k --fts_dropout --prob unet
./adsb3_cache_ft_batch.py --bits 8 --fts ft_k --fts_dropout --prob unet --mode 3

exit
./adsb3_cache_ft_batch.py --bits 8 --prob unet --mode 4
./adsb3_cache_ft_batch.py --bits 8 --fts ft_z --fts_dropout --prob unet

exit
./adsb3_cache_ft_batch.py --bits 8 --prob unet_k --mask hull

exit

./adsb3_cache_ft_batch.py --bits 8 --fts ft_k --fts_dropout --prob unet_k --mask hull
exit
./adsb3_cache_ft_batch.py --bits 8 --prob unet_k --mask hull
exit

./adsb3_cache_ft_batch.py --bits 8 --prob none_k
./adsb3_cache_ft_batch.py --bits 8 --prob small_k
exit
./adsb3_cache_ft_batch.py --bits 8 --prob vnet
exit
#./adsb3_cache_ft_batch.py --bits 8 --fts ft_k --fts_dropout
./adsb3_cache_ft_batch.py --bits 8 --prob tiny_k
exit
SPACING=0.6 ./adsb3_cache_ft_batch.py --bits 8 --prob unet_k
exit
./adsb3_cache_ft_batch.py --bits 8 --prob unet_k
./adsb3_cache_ft_batch.py --bits 8 --prob unet_l
exit
./adsb3_cache_ft_batch.py --bits 8 --prob lymph_unet
SPACING=0.6 ./adsb3_cache_ft_batch.py --bits 8 --prob unet 
exit
./adsb3_cache_ft_batch.py --bits 8 --prob unet 

SPACING=1.2 ./adsb3_cache_ft.py --bits 8 --prob luna_tiny
exit
SPACING=1.2 ./adsb3_cache_ft.py --bits 8 --prob luna
exit
SPACING=1.2 ./adsb3_cache_ft.py --bits 8 --prob luna_dilate
./adsb3_cache_ft.py --bits 8
./adsb3_cache_ft.py --bits 8 --mode 3
./adsb3_cache_ft.py --bits 8 --fts tcia.ft
./adsb3_cache_ft.py --bits 8 --prob luna.ns --channels 1
./adsb3_cache_ft.py --bits 8 --prob tcia.none
./adsb3_cache_ft.py --bits 8 --prob tcia.small
./adsb3_cache_ft.py --bits 8 --prob nnc
#./adsb3_cache_ft.py --prob lymph3c16
./adsb3_cache_ft.py --bits 8 --fts ac --stride 32

exit
./adsb3_cache_ft.py --bits 8 --fts tcia.ft.X
exit
