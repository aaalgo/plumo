#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append('build/lib.linux-x86_64-2.7')
import shutil
import math
from glob import glob
import cv2
import csv
import random
from PIL import Image, ImageDraw
import dicom
import copy
import numpy as np
import SimpleITK as itk
from skimage import measure
import logging
import cPickle as pickle
import plumo

# configuration options

DICOM_STRICT = False
SPACING = 0.8
GAP = 5
FAST = 400

if 'SPACING' in os.environ:
    SPACING = float(os.environ['SPACING'])
    print('OVERRIDING SPACING = %f' % SPACING)

if 'GAP' in os.environ:
    GAP = int(os.environ['GAP'])
    print('OVERRIDING GAP = %d' % GAP)


ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, 'data', 'adsb3')

# parse label file, return a list of (id, label)
# if gs is True, labels are int
# otherwise, labels are float
def load_meta (path, gs=True):
    all = []
    with open(path, 'r') as f:
        header = f.next()    # skip one line
        assert header.strip() == 'id,cancer'
        for l in f:
            ID, label = l.strip().split(',')
            if gs:
                label = int(label)
            else:
                label = float(label)
            all.append((ID, label))
            pass
        pass
    return all

# write meta for verification
def dump_meta (path, meta):
    with open(path, 'w') as f:
        f.write('id,cancer\n')
        for ID, label in meta:
            f.write('%s,%s\n' % (ID, str(label)))
            pass
        pass


STAGE1_TRAIN = load_meta(os.path.join(DATA_DIR, 'stage1_labels.csv'))
STAGE1_PUBLIC = load_meta(os.path.join(DATA_DIR, 'stage1_public.csv'))
STAGE2_PUBLIC = load_meta(os.path.join(DATA_DIR, 'stage2_public.csv'))
STAGE2_PRIVATE = load_meta(os.path.join(DATA_DIR, 'stage2_private.csv'))

ALL_CASES = STAGE1_TRAIN + STAGE1_PUBLIC + STAGE2_PUBLIC + STAGE2_PRIVATE


# All DiCOMs of a UID, organized
class Case (plumo.DicomVolume):
    def __init__ (self, uid, regroup = True):
        path = os.path.join(DATA_DIR, 'dicom', uid)
        plumo.DicomVolume.__init__(self, path)
        self.uid = uid
        self.path = path
        pass
    pass


def save_mask (path, mask):
    shape = np.array(list(mask.shape), dtype=np.uint32)
    total = mask.size
    totalx = (total +7 )// 8 * 8
    if totalx == total:
        padded = mask
    else:
        padded = np.zeros((totalx,), dtype=np.uint8)
        padded[:total] = np.reshape(mask, (total,))
        pass
    padded = np.reshape(padded, (totalx//8, 8))
    #print padded.shape
    packed = np.packbits(padded)
    #print packed.shape
    np.savez_compressed(path, shape, packed)
    pass

def load_mask (path):
    import sys
    saved = np.load(path)
    shape = saved['arr_0']
    D, H, W = shape
    size = D * H * W
    packed = saved['arr_1']
    padded = np.unpackbits(packed)
    binary = padded[:size]
    return np.reshape(binary, [D, H, W])

def load_8bit_lungs_noseg (uid):
    case = Case(uid)
    case.normalize_8bit()
    return case

def load_16bit_lungs_noseg (uid):
    case = Case(uid)
    case.normalize_16bit()
    return case

def load_lungs_mask (uid):
    cache = os.path.join('maskcache/mask-v2/%s.npz' % case.uid)
    binary = None
    if os.path.exists(cache) and os.path.getsize(cache) > 0:
        # load cache
        binary = load_mask(cache)
        assert not binary is None
    if binary is None:
        case = load_case(uid)
        case.normalizeHU()
        binary = segment_lung_axial_v2(case.images) #, th=200.85)
        save_mask(cache, binary)
        pass
    return binary

def load_fts (path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    pass


def patch_clip_range (x, tx, wx, X):
    if x < 0:   #
        wx += x
        tx -= x
        x = 0
    if x + wx > X:
        d = x + wx - X
        wx -= d
        pass
    return x, tx, wx

def extract_patch_3c (images, z, y, x, size):
    assert len(images.shape) == 3
    _, Y, X = images.shape
    z = int(round(z))
    y = int(round(y))
    x = int(round(x))
    image = get3c(images, z)
    if image is None:
        return None
    ty = 0
    tx = 0
    y -= size//2
    x -= size//2
    wy = size
    wx = size
    #print y, ty, wy, x, tx, wx
    y, ty, wy = patch_clip_range(y, ty, wy, Y)
    x, tx, wx = patch_clip_range(x, tx, wx, X)
    # now do overlap
    patch = np.zeros((size, size, 3), dtype=image.dtype)
    #print y, ty, wy, x, tx, wx
    patch[ty:(ty+wy),tx:(tx+wx),:] = image[y:(y+wy),x:(x+wx),:]
    return patch

def try_mkdir (path):
    try:
        os.makedirs(path)
    except:
        pass

def try_remove (path):
    try:
        os.remove(path)
    except:
        shutil.rmtree(path, ignore_errors=True)
    pass

if __name__ == '__main__':
    #dump_meta('a', STAGE1.train)
    #dump_meta('b', STAGE1.test)
    pass
