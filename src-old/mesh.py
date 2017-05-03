#!/usr/bin/env python
import sys
import time
import numpy as np
import cv2
import skimage
from skimage import measure
from scipy.ndimage.morphology import grey_dilation, binary_dilation, binary_fill_holes
#from skimage import regionprops
from adsb3 import *
import scipy
import pyadsb3

def pad (images, padding=2, dtype=None):
    Z, Y, X = images.shape
    if dtype is None:
        dtype = images.dtype
    out = np.zeros((Z+padding*2, Y+padding*2, X+padding*2), dtype=dtype)
    out[padding:(Z+padding),padding:(Y+padding),padding:(X+padding)] = images
    return out

def segment_body (image, smooth=1, th=-300):
    blur = scipy.ndimage.filters.gaussian_filter(image, smooth, mode='constant')
    binary = np.array(blur < th, dtype=np.uint8)

    # body is a rough region covering human body
    body = np.zeros_like(binary)
    for i, sl in enumerate(binary):
        #H, W = sl.shape
        ll = measure.label(sl, background=1)   # connected components
        # biggest CC should be body
        pp = measure.regionprops(ll)
        boxes = [(x.area, x.bbox, x.filled_image) for x in pp if x.label != 0]  # label 0 is air
        boxes = sorted(boxes, key = lambda x: -x[0])
        if len(boxes) == 0:
            continue
        y0, x0, y1, x1 = boxes[0][1]
        body[i,y0:y1,x0:x1] = boxes[0][2]
        pass
    return body, None

def fill_convex (image):
    H, W = image.shape
    padded = np.zeros((H+20, W+20), dtype=np.uint8)
    padded[10:(10+H),10:(10+W)] = image

    contours = measure.find_contours(padded, 0.5)
    if len(contours) == 0:
        return image
    if len(contours) == 1:
        contour = contours[0]
    else:
        contour = np.vstack(contours)
    cc = np.zeros_like(contour, dtype=np.int32)
    cc[:,0] = contour[:, 1]
    cc[:,1] = contour[:, 0]
    hull = cv2.convexHull(cc)
    contour = hull.reshape((1, -1, 2)) 
    cv2.fillPoly(padded, contour, 1)
    return padded[10:(10+H),10:(10+W)]

def segment_lung (image, smooth=1, th=-300):

    padding_value = np.min(image)
    if padding_value < -1010:
        padding = [image == padding_value]
    else:
        padding = None

    imagex = image
    if padding:
        imagex = np.copy(image) 
        imagex[padding] = 0
    blur = scipy.ndimage.filters.gaussian_filter(imagex, smooth, mode='constant')
    if padding:
        blur[padding] = padding_value

    binary = np.array(blur < th, dtype=np.uint8)


    # body is a rough region covering human body
    body = np.zeros_like(binary)

    for i, sl in enumerate(binary):
        #H, W = sl.shape
        ll = measure.label(sl, background=1)   # connected components
        # biggest CC should be body
        pp = measure.regionprops(ll)
        boxes = [(x.area, x.bbox, x.filled_image) for x in pp if x.label != 0]  # label 0 is air

        boxes = sorted(boxes, key = lambda x: -x[0])
        if len(boxes) == 0:
            print 'no body detected'
            continue
        y0, x0, y1, x1 = boxes[0][1]
        body[i,y0:y1,x0:x1] = fill_convex(boxes[0][2])
        pass

    binary *= body

    if False:
        padding = np.min(image)
        if padding < -1010:
            binary[image == padding] = 0

        # 0: body
        # 1: air & lung

        labels = measure.label(binary, background=1)

        # set air (same cc as corners) -> body
        bg_labels = set()
        # 8 corders of the image
        for z in [0, -1]:
            for y in [0, -1]:
                for x in [0, -1]:
                    bg_labels.add(labels[z, y, x])
        print bg_labels
        bg_labels = list(bg_labels)
        for bg_label in bg_labels:
            binary[bg_label == labels] = 0
            pass

    # now binary:
    #   0: non-lung & body tissue in lung & air
    #   1: lung & holes in body
    #inside = np.copy(binary)


    # now binary:
    #   0: non-lung & body tissue in lung
    #   1: lung & holes in body
    binary = np.swapaxes(binary, 0, 1)
    for i, sl in enumerate(binary):
        #H, W = sl.shape
        ll = measure.label(sl, background=1)   # connected components
        # biggest CC should be body
        vv, cc = np.unique(ll, return_counts=True) 
        cc[0] = 0
        assert len(vv) > 0
        body_ll = vv[np.argmax(cc)]
        binary[i][ll != body_ll] = 1
        pass
    binary = np.swapaxes(binary, 0, 1)
    if padding:
        binary[padding] = 0
    binary *= body

    # binary    0: body
    #           1: - anything inside lung
    #              - holes in body
    #              - possibly image corners
    #

    # inside    0: non-lung & air
    #              body tissue in lung
    #           1: lung

    # set corner again
    labels = measure.label(binary, background=0)
    bg_labels = set([0])
    for z in [0, -1]:
        for y in [0, -1]:
            for x in [0, -1]:
                bg_labels.add(labels[z, y, x]) 

	#print 'bg', bg_labels
    val_counts = zip(*np.unique(labels, return_counts=True))
    val_counts = [x for x in val_counts if (not x[0] in bg_labels) and (x[1] >= 10)]
    val_counts = sorted(val_counts, key=lambda x:-x[1])[:100] # sort by size
    body_counts = [c for _, c in val_counts]
    print val_counts
    binary = np.zeros_like(binary, dtype=np.uint8)
    print val_counts[0][0]
    binary[labels == val_counts[0][0]] = 1
    #for v, _ in val_counts[0:5]:
    #    binary[labels == v] = 1
    if len(val_counts) > 1:
        if val_counts[1][1] * 3 > val_counts[0][1]:
            #binary[labels == val_counts[1][0]] = 1
            #if val_counts[1][1] * 4 > val_counts[0][1]:
            logging.warn('more than 2 lungs parts detected')

    # remove upper part of qiguan
    last = binary.shape[0] - 1
    for ri in range(binary.shape[0]):
        #H, W = sl.shape
        i = last - ri
        ll = measure.label(binary[i], background=0)   # connected components
        nl = np.unique(ll)
        if len(nl) <= 2:
            binary[i,:,:] = 0
        else:
            print 'removed %d slices' % ri
            break
        pass

    return binary, body_counts #, inside

def convex_hull (binary):
    swap_sequence = [(0, 1),  # 102
                     (0, 2),  # 201
                     (0, 2)]  # 102

    output = np.ndarray(binary.shape, dtype=binary.dtype)
    for swp1, swp2 in swap_sequence:
        N = binary.shape[0]
        print 'shape', binary.shape
        for i in range(N):
            contours = measure.find_contours(binary[i], 0.5)
            if len(contours) == 0:
                continue
            if len(contours) == 1:
                contour = contours[0]
            else:
                contour = np.vstack(contours)
            cc = np.zeros_like(contour, dtype=np.int32)
            cc[:,0] = contour[:, 1]
            cc[:,1] = contour[:, 0]
            hull = cv2.convexHull(cc)
            contour = hull.reshape((1, -1, 2)) 
            cv2.fillPoly(binary[i], contour, 1)
            #binary[i] = skimage.morphology.convex_hull_image(binary[i])
            pass
        print 'swap', swp1, swp2
        nb = np.swapaxes(binary, swp1, swp2)
        binary = np.ndarray(nb.shape, dtype=nb.dtype)
        binary[:,:] = nb[:,:]
        pass
    binary = np.swapaxes(binary, 0, 1)
    output[:,:] = binary[:,:]
    return output;
    #binary = binary_dilation(output, iterations=dilate)
    #return binary

def segment_lung_internal (image, smooth=1, th=-300):

    padding_value = np.min(image)
    if padding_value < -1010:
        padding = [image == padding_value]
    else:
        padding = None

    imagex = image
    if padding:
        imagex = np.copy(image) 
        imagex[padding] = 0
    blur = scipy.ndimage.filters.gaussian_filter(imagex, smooth, mode='constant')
    if padding:
        blur[padding] = padding_value

    binary = np.array(blur < th, dtype=np.uint8)

    #not_slid = np.array(blur < th, dtype=np.uint8)
    not_solid = np.copy(binary)


    # body is a rough region covering human body
    body = np.zeros_like(binary)

    for i, sl in enumerate(binary):
        #H, W = sl.shape
        ll = measure.label(sl, background=1)   # connected components
        # biggest CC should be body
        pp = measure.regionprops(ll)
        boxes = [(x.area, x.bbox, x.filled_image) for x in pp if x.label != 0]  # label 0 is air

        boxes = sorted(boxes, key = lambda x: -x[0])
        if len(boxes) == 0:
            print 'no body detected'
            continue
        y0, x0, y1, x1 = boxes[0][1]
        body[i,y0:y1,x0:x1] = fill_convex(boxes[0][2])
        pass

    binary *= body

    if False:
        padding = np.min(image)
        if padding < -1010:
            binary[image == padding] = 0

        # 0: body
        # 1: air & lung

        labels = measure.label(binary, background=1)

        # set air (same cc as corners) -> body
        bg_labels = set()
        # 8 corders of the image
        for z in [0, -1]:
            for y in [0, -1]:
                for x in [0, -1]:
                    bg_labels.add(labels[z, y, x])
        print bg_labels
        bg_labels = list(bg_labels)
        for bg_label in bg_labels:
            binary[bg_label == labels] = 0
            pass

    # now binary:
    #   0: non-lung & body tissue in lung & air
    #   1: lung & holes in body
    #inside = np.copy(binary)


    # now binary:
    #   0: non-lung & body tissue in lung
    #   1: lung & holes in body
    binary = np.swapaxes(binary, 0, 1)
    for i, sl in enumerate(binary):
        #H, W = sl.shape
        ll = measure.label(sl, background=1)   # connected components
        # biggest CC should be body
        vv, cc = np.unique(ll, return_counts=True) 
        cc[0] = 0
        assert len(vv) > 0
        body_ll = vv[np.argmax(cc)]
        binary[i][ll != body_ll] = 1
        pass
    binary = np.swapaxes(binary, 0, 1)
    if padding:
        binary[padding] = 0
    binary *= body

    # binary    0: body
    #           1: - anything inside lung
    #              - holes in body
    #              - possibly image corners
    #

    # inside    0: non-lung & air
    #              body tissue in lung
    #           1: lung

    # set corner again
    labels = measure.label(binary, background=0)
    bg_labels = set([0])
    for z in [0, -1]:
        for y in [0, -1]:
            for x in [0, -1]:
                bg_labels.add(labels[z, y, x]) 

	#print 'bg', bg_labels
    val_counts = zip(*np.unique(labels, return_counts=True))
    val_counts = [x for x in val_counts if (not x[0] in bg_labels) and (x[1] >= 10)]
    val_counts = sorted(val_counts, key=lambda x:-x[1])[:100] # sort by size
    body_counts = [c for _, c in val_counts]
    print val_counts
    binary = np.zeros_like(binary, dtype=np.uint8)
    print val_counts[0][0]
    binary[labels == val_counts[0][0]] = 1
    #for v, _ in val_counts[0:5]:
    #    binary[labels == v] = 1
    if len(val_counts) > 1:
        if val_counts[1][1] * 3 > val_counts[0][1]:
            #binary[labels == val_counts[1][0]] = 1
            #if val_counts[1][1] * 4 > val_counts[0][1]:
            logging.warn('more than 2 lungs parts detected')

    # remove upper part of qiguan
    last = binary.shape[0] - 1
    for ri in range(binary.shape[0]):
        #H, W = sl.shape
        i = last - ri
        ll = measure.label(binary[i], background=0)   # connected components
        nl = np.unique(ll)
        if len(nl) <= 2:
            binary[i,:,:] = 0
        else:
            print 'removed %d slices' % ri
            break
        pass

    #not_solid = np.logical_and(not_solid, binary)   # solid within lung 
    return np.logical_and(not_solid, binary), body_counts #, inside

