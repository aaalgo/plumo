#!/usr/bin/env python
import os
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

# configuration options

DICOM_STRICT = False
SPACING = 0.8
GAP = 5
FAST = 400

if 'SPACING' in os.environ:
    SPACING = float(os.environ['SPACING'])
    print 'OVERRIDING SPACING = %f' % SPACING

if 'GAP' in os.environ:
    GAP = int(os.environ['GAP'])
    print 'OVERRIDING GAP = %d' % GAP

def get3c (images, i):
    if i < GAP:
        return None
    if i + GAP >= images.shape[0]:
        return None
    a = images[i-GAP]
    b = images[i]
    c = images[i+GAP]
    c3 = np.zeros(a.shape + (3,), dtype=np.float32)
    c3[:,:,0] = a
    c3[:,:,1] = b
    c3[:,:,2] = c
    return c3


#######################

def trim_loc (array1d, margin=0):
    w = np.where(array1d > 0)
    x0 = np.min(w)
    x1 = np.max(w)+1
    return max(0, x0-margin), min(x1+margin, array1d.shape[0])

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

def chunks (l, n):
    for i in range(0, len(l), n):
        yield l[i:(i+n)]

ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, 'data')

def dicom_error (dcm, msg, level=logging.ERROR):
    s = 'DICOM ERROR (%s): %s' % (dcm.filename, msg)
    if DICOM_STRICT and level >= logging.ERROR:
        raise Exception(s)
    else:
        logging.log(level, s)
    pass

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
            #print ID, label
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

# stores example uid & label information
# Stage.train = [(uid, label)]
# Stage.test = [(uid, 0.5)]
# Stage.examples = [(uid, 0.5)]
class Bowl:
    def __init__ (self):
        self.train1 = load_meta(os.path.join(DATA_DIR, 'stage1_labels.csv'))
        self.train = load_meta(os.path.join(DATA_DIR, 'train.csv'))
        self.test = load_meta(os.path.join(DATA_DIR, 'test.csv'), gs=False)
        pass
    pass

#BOWL = Bowl()

#UIDS = set()
#UIDS.update([uid for uid, _ in BOWL.train])
#UIDS.update([uid for uid, _ in BOWL.test])

def dcm_sanity_check (dcm):
    rx, ry, rz, cx, cy, cz = [float(v) for v in dcm.ImageOrientationPatient]
    pass

class DICOM:
    def __init__ (self, dcm):
        self.patient_id = dcm.PatientID
        self.study_id = dcm.StudyInstanceUID
        self.series_id = dcm.SeriesInstanceUID
        self.HU = (float(dcm.RescaleSlope), float(dcm.RescaleIntercept))
        # filename as slice ID
        self.sid = os.path.splitext(os.path.basename(dcm.filename))[0]
        self.dcm = dcm
        self.image = dcm.pixel_array
        self.shape = dcm.pixel_array.shape
        self.pixel_padding = None
        try:
            self.pixel_padding = int(dcm.PixelPaddingValue)
        except:
            pass
        from dicom.tag import Tag
        #tag = Tag(0x0020,0x0032)
        #print dcm[tag].value
        #print dcm.ImagePositionPatient
        #assert dcm[tag] == dcm.ImagePositionPatient
        x, y, z = [float(v) for v in dcm.ImagePositionPatient]
        self.position = (x, y, z)
        rx, ry, rz, cx, cy, cz = [float(v) for v in dcm.ImageOrientationPatient]
        self.ori_row = (rx, ry, rz)
        self.ori_col = (cx, cy, cz)
        x, y = [float(v) for v in dcm.PixelSpacing]
        assert x == y
        self.spacing = x
        # Stage1: 4704 missing SliceLocation
        try:
            self.location = float(dcm.SliceLocation)
        except:
            dicom_error(dcm, 'Missing SliceLocation', level=logging.DEBUG)
            self.location = self.position[2]
            pass
        self.bits = dcm.BitsStored


        if False:
            # Non have SliceThickness
            tag = Tag(0x0018, 0x0050)
            if not tag in dcm:
                dicom_error(dcm, 'Missing SliceThickness', level=logging.WARN)
            else:
                logging.info('Has SliceThickness: %s' % dcm.filename)
                self.thickness = float(dcm[tag].value)

        # ???, why the value is as big as 63536
        if False:
            # Stage1 data:
            # 4704 have padding, 126057 not, so skip this
            self.padding = None
            try:
                self.padding = dcm.PixelPaddingValue
            except:
                dicom_error(dcm, 'Missing PixelPaddingValue', level=logging.WARN)
                pass

        # sanity check
        #if dcm.PatientName != dcm.PatientID:
        #    dicom_error(dcm, 'PatientName is not dcm.PatientID')
        if dcm.Modality != 'CT':
            dicom_error(dcm, 'Bad Modality: ' + dcm.Modality)
        #if Tag(0x0008,0x103e) in dcm:
        if False:
            if dcm.SeriesDescription != 'Axial' and dcm.SeriesDescription != 'mediastinal_lymph_nodes' and dcm.SeriesDescription != 'Recon 2: ACRIN LARGE' and dcm.SeriesDescription != 'Recon 3: CHEST-ABD':
                dicom_error(dcm, 'Bad SeriesDescription: ' + dcm.SeriesDescription)
        #if Tag(0x0008,0x0008) in dcm:
        #    if not 'AXIAL' in ' '.join(list(dcm.ImageType)).upper():
        #        dicom_error(dcm, 'Bad image type: ' + list(dcm.ImageType))

        ori_type_tag = Tag(0x0010,0x2210)
        if ori_type_tag in dcm:
            ori_type = dcm[ori_type_tag].value
            if 'BIPED' != ori_type:
                dicom_error(dcm, 'Bad Anatomical Orientation Type: ' + ori_type)
        # location should roughly be position.z
        self.funny_slice_location = abs(self.position[2] - self.location) > 10

        x, y, z = self.ori_row  # should be (1, 0, 0)
        if x < 0.9:
            dicom_error(dcm, 'Bad row orientation')
        x, y, z = self.ori_col  # should be (0, 1, 0)
        if y < 0.9:
            dicom_error(dcm, 'Bad col orientation')
        pass
    pass

def segment_lung_axial (image, th=123.85, dilate=0.01):

    blur = np.copy(image)
    for i in range(blur.shape[0]):
        cv2.blur(blur[i], (5,5), blur[i])

    binary = np.array(blur < th, dtype=np.uint8)
    # 0: body
    # 1: air & lung
    labels = measure.label(binary, background=-1)

    # set air (same cc as corners) -> body
    bg_labels = set()
    for z in [0, -1]:
        for y in [0, -1]:
            for x in [0, -1]:
                bg_labels.add(labels[z, y, x])
    bg_labels = list(bg_labels)
    print(bg_labels)
    if len(bg_labels) > 1:
        logging.warn('bg not connected, detected %d components' % len(bg_labels))
        pass
    for bg_label in bg_labels:
        binary[bg_label == labels] = 0
        pass


    # now binary:
    #   0: non-lung & body tissue in lung
    #   1: lung & holes in body
    for i, sl in enumerate(binary):
        #H, W = sl.shape
        ll = measure.label(sl, background=-1)   # connected components
        # biggest CC should be body
        vv, cc = np.unique(ll, return_counts=True) 
        assert len(vv) > 0
        body_ll = vv[np.argmax(cc)]
        binary[i][ll != body_ll] = 1
        pass

    # set corner again
    labels = measure.label(binary, background=0)
    bg_labels = set([0])
    for z in [0, -1]:
        for y in [0, -1]:
            for x in [0, -1]:
                bg_labels.add(labels[z, y, x])

    val_counts = zip(*np.unique(labels, return_counts=True))
    val_counts = [x for x in val_counts if not x[0] in bg_labels]   # remove background
    val_counts = sorted(val_counts, key=lambda x:-x[1]) # sort by size
    th = val_counts[0][1] /4    # 1/4 size of the larged connected component (must be lung)
    val = [v for v, c in val_counts if c >= th]
    if len(val) >= 3:
        logging.warn('more than 2 lungs parts detected %d' % len(val))
    binary = np.zeros_like(binary, dtype=np.uint8)
    for v in val:
        binary[labels == v] = 1

    H, W = binary[0].shape
    dilate = int(round(math.sqrt(1.0 * H * W) * dilate))
    #print("DILATE: ", dilate)
    kernel = np.ones((dilate, dilate), dtype=np.int32)

    for i in range(binary.shape[0]):
        cv2.dilate(binary[i], kernel, binary[i])

    #image[binary == 0] = 255
    #image = 255 - image
    #image[binary == 0] = 255
    return binary
    #return image #* binary.astype(image.dtype)


AXIAL = 0
SAGITTAL = 1
CORONAL = 2

VIEWS = [AXIAL, SAGITTAL, CORONAL]
VIEW_NAMES = ['axial', 'sagittal', 'coronal']

AXES_ORDERS = ([0, 1, 2],  # AXIAL
               [2, 1, 0],  # SAGITTAL
               [1, 0, 2])  # CORONAL


def index_view (I, view):
    assert len(I) == 3
    a, b, c = AXES_ORDERS[view]
    return [I[a], I[b], I[c]]

def strip_pad_512 (n, size=512):
    if n >= size:
        from_x = (n-size)/2
        to_x = 0
        n_x = size
        shift_x = from_x
    else:
        from_x = 0
        to_x = (size-n)/2
        n_x = n
        shift_x = -to_x
    return from_x, to_x, n_x, shift_x

class CaseBase (object):
    # self.images
    # self.spacing
    # self.origin   # !!! origin is never transposed!!!
    # self.axes
    # self.vspacing
    def __init__ (self):
        self.uid = None
        self.path = None
        self.images = None      # 3-D array
        self.spacing = None     #
        self.origin = None      # origin never changes
                                # under transposing
        self.view = None
        self.anno = None
        # We save the coefficients for normalize to
        # Hounsfield Units, and keep that updated
        # when normalizing
        self.HU = None          # (intercept, slope)
        self.dcm_z_position = None

        self.orig_origin = None
        self.orig_spacing = None
        self.orig_shape = None
        self.pixel_padding = None
        pass

    def copy_replace_images (self, images):
        case = CaseBase()
        case.uid = self.uid
        case.orig_origin = self.orig_origin
        case.orig_spacing = self.orig_spacing
        case.orig_shape = self.orig_shape
        case.path = self.path
        case.images = images
        case.spacing = self.spacing
        case.view = self.view
        case.origin = self.origin
        case.anno = self.anno
        return case

    def normalizeHU (self):
        assert not self.HU is None
        a, b = self.HU
        self.images *= a
        self.images += b
        self.HU = (1.0, 0)
        if not self.pixel_padding is None:
            self.pixel_padding = self.pixel_padding * a + b
        pass

    def transpose_array (self, view, array):
        if self.view == view:
            return array
        elif self.view == AXIAL and view == SAGITTAL:
            d1, d2 = 0, 2
        elif self.view == AXIAL and view == CORONAL:
            d1, d2 = 0, 1
        elif self.view == SAGITTAL and view == AXIAL:
            d1, d2 = 0, 2
        elif self.view == CORONAL and view == AXIAL:
            d1, d2 = 0, 1
        else:
            assert False
        return np.swapaxes(array, d1, d2)

    def transpose (self, view):
        if self.view == view:
            return self
        elif self.view == AXIAL and view == SAGITTAL:
            d1, d2 = 0, 2
        elif self.view == AXIAL and view == CORONAL:
            d1, d2 = 0, 1
        elif self.view == SAGITTAL and view == AXIAL:
            d1, d2 = 0, 2
        elif self.view == CORONAL and view == AXIAL:
            d1, d2 = 0, 1
        else:
            assert False

        case = CaseBase()
        case.uid = self.uid
        case.orig_origin = self.orig_origin
        case.orig_spacing = self.orig_spacing
        case.orig_shape = self.orig_shape
        case.path = self.path
        case.images = np.swapaxes(self.images, d1, d2)
        assert isinstance(self.spacing, tuple)
        sp = list(self.spacing)
        sp[d1], sp[d2] = sp[d2], sp[d1]
        case.spacing = tuple(sp)
        case.view = view
        case.origin = self.origin
        case.anno = self.anno
        return case

    def round512 (self, size=512):
        target = np.zeros((size,size,size), dtype=self.images.dtype)
        Z, Y, X = self.images.shape
        from_z, to_z, n_z, shift_z = strip_pad_512(Z, size=size)
        from_y, to_y, n_y, shift_y = strip_pad_512(Y, size=size)
        from_x, to_x, n_x, shift_x = strip_pad_512(X, size=size)
        target[to_z:(to_z+n_z),
               to_y:(to_y+n_y),
               to_x:(to_x+n_x)] = self.images[from_z:(from_z+n_z),
                   from_y:(from_y+n_y),
                   from_x:(from_x+n_x)]
        self.origin[0] += shift_z * self.spacing[0]
        self.origin[1] += shift_y * self.spacing[1]
        self.origin[2] += shift_x * self.spacing[2]
        print("off", to_x, to_y, to_z)
        print("len", n_x, n_y, n_z)
        print("shi", shift_x, shift_y, shift_z)
        self.images = target
        pass

    def strip (self, mask, margin1=2, margin2=10):
        z0, z1 = trim_loc(np.sum(mask, axis=(1,2)), margin=margin1)
        y0, y1 = trim_loc(np.sum(mask, axis=(0,2)), margin=margin2)
        x0, x1 = trim_loc(np.sum(mask, axis=(0,1)), margin=margin2)
        self.origin[0] += z0 * self.spacing[0]
        self.origin[1] += y0 * self.spacing[1]
        self.origin[2] += x0 * self.spacing[2]
        self.images = self.images[z0:z1, y0:y1, x0:x1]
        pass

    def round_stride (self, stride=16):
        T, H, W = self.images.shape[:3]
        nT = T / stride * stride
        nH = H / stride * stride
        nW = W / stride * stride
        oT = (T - nT)/2
        oH = (H - nH)/2
        oW = (W - nW)/2
        self.origin[0] += oT * self.spacing[0]
        self.origin[1] += oH * self.spacing[1]
        self.origin[2] += oW * self.spacing[2]
        self.images = self.images[oT:(oT+nT),oH:(oH+nH),oW:(oW+nW)]
        return oT, oH, oW
        pass

    # consider using scipy.ndimage.interpolation
    def rescale (self, slices = None, spacing = None, size = None, method=2):
        # if slices != self.images.shape[0], use method:
        #   0:  adjust slices, so everything is integer and no rounding or approx. is done
        #   1:  do not change slices, use nearest neighbor
        #   2:  do not change slices, use interpolation
        N, H, W = self.images.shape
        case = CaseBase()
        case.uid = self.uid
        case.orig_origin = self.orig_origin
        case.orig_spacing = self.orig_spacing
        case.orig_shape = self.orig_shape
        case.path = self.path
        case.view = self.view
        case.origin = self.origin
        case.anno = self.anno
        case.HU = self.HU
        assert (spacing and not size) or (size and not spacing)
        if (not slices) or (slices == N):
            method = 0
            slices = N
            step = 1
            off = 0
            sp1 = self.spacing[0]
        elif method == 0:   # TODO: need to do actual samping
                # origin under this is not correct due to non-0 off
            step = int(round(N / slices))
            slices = N / step
            off = (N - slices * step) / 2
            sp1 = self.spacing[0] * step
        else:
            off = 0
            step = float(N -1)/ (slices - 1)
            sp1 = self.spacing[0] * step
            pass
        if spacing:
            H = int(round((H-1) * self.spacing[1] / spacing + 1))
            W = int(round((W-1) * self.spacing[2] / spacing + 1))
            resize = (W, H)
            sp2 = spacing
            sp3 = spacing
        elif size:
            sp2 = self.spacing[1] * (H-1) / (size-1)
            sp3 = self.spacing[2] * (W-1) / (size-1)
            resize = (size, size)
            H = size
            W = size
        else:
            resize = None
            _, sp2, sp3 = self.spacing

        case.spacing = (sp1, sp2, sp3)
        case.images = np.zeros((slices, H, W), dtype=np.float32)

        for i in range(slices):
            if method == 0 or method == 1:
                arr = int(round(off))
                image = self.images[arr, :, :]
            elif method == 2:
                L = int(math.floor(off))
                R = int(math.ceil(off))
                if R <= 0:
                    image = self.images[0, :, :]
                elif L >= N-1:
                    image = self.images[N-1, :, :]
                elif R - L < 0.5:   # R == L
                    image = self.images[L, :, :]
                else:
                    image = (self.images[L, :, :]  * (R - off) + self.images[R, :, :] * (off - L)) / (R - L)
                pass

            if resize:
                cv2.resize(image, resize, case.images[i, :, :])
            else:
                case.images[i, :, :] = image
            off += step

        return case

    def rescale3D (self, spacing):
        slices = int(round(self.spacing[0] * (self.images.shape[0] - 1) / spacing + 1))
        return self.rescale(slices, spacing, size=None, method=2)
    pass

    def normalize (self, min=0, max=1, min_th = -1000, max_th = 400):
        assert self.images.dtype == np.float32
        if not min_th is None:
            self.images[self.images < min_th] = min_th
        if not max_th is None:
            self.images[self.images > max_th] = max_th
        m = min_th #np.min(self.images)
        M = max_th #np.max(self.images)
        scale = (1.0 * max - min)/(M - m)
        logging.debug('norm %f %f' % (m, M))
        self.images -= m
        self.images *= scale
        self.images += min
        # recalculate HU 
        #   I:  original image
        #   I': new image
        #   a'I' + b' = aI + b
        #   I' = (I-m) * scale + min
        #      = I*scale + (min - m * scale)
        #   so
        #   a'I*scale + (min - m * scale)*a' + b' = aI + b
        #   
        #   a' = a / scale
        #   b' = b + a'(m * scale -min)
        #      = b + a * (m - min/scale)
        if self.HU:
            a, b = self.HU
            #self.HU = (a * (M -m), b + a * m)
            self.HU = (a / scale, b + a * (m - min/scale))
        pass

    def standardize_color (self):
        self.normalizeHU()
        self.normalize(min_th=-1000,max_th=400,min=0,max=255)
        pass

    def standardize_color16 (self):
        self.normalizeHU()
        self.normalize(min_th=-1000,max_th=400,min=0,max=1400)
        pass

    # return center coordinate
    def world_to_vox (self, world):
        # change view
        # change origin
        z, y, x, r = world
        z0, y0, x0 = self.origin
        cc = (np.array(world[:3])-np.array(self.origin))
        cc = cc[AXES_ORDERS[self.view]]
        spacing = np.array(self.spacing)
        cc = cc / spacing
        rr = r / spacing
        #print "xxx", cc[0], rr[0]
        return  cc, rr

    def picpac_anno (self):
        # !!! annotation is center & radius instead of orign + size in picpac
        if self.anno is None:
            return []
        ALL = []
        nodules = [ self.world_to_vox(world) for world in self.anno]
        C, H, W = self.images.shape
        for (z, y, x), (zr, yr, xr) in nodules:
            first = max(0, int(math.ceil(z - zr)))
            last = min(C-1, int(math.floor(z + zr)))
            if first > last:
                continue
            nod = []
            x /= W
            y /= H 
            for i in range(first, last + 1):
                cos = abs(i - z) / zr
                sin = math.sqrt(1 - cos * cos)
                cyr = yr * sin / H
                cxr = xr * sin / W
                nod.append([i, x, y, cxr, cyr])
                #print 'ellipse', x, y, xr, yr
                #pass
                pass
            ALL.append(nod)
            pass
        return ALL

    def papaya_box (self, box):
        out = [0]*6
        assert self.view == AXIAL
        for i in range(3):
            out[i] = int(round((self.origin[i] + self.spacing[i] * box[i] - self.orig_origin[i]) / self.orig_spacing[i]))
            out[i+3] = int(round((self.origin[i] + self.spacing[i] * box[i+3] - self.orig_origin[i]) / self.orig_spacing[i]))
            pass
        D, _, W = self.orig_shape
        out[0], out[3] = D-out[3], D-out[0]
        out[2], out[5] = W-out[5], W-out[2]
        return out

    def save_gif (self, path, anno=False, aug=2, step=1):
        # must normalize first to [0, 1]
        cube = np.uint8(np.clip(self.images, 0, 255))
        frames = [Image.fromarray(cube[i,:,:]) for i in range(0, cube.shape[0], step)]
        if anno:
            C, H, W = self.images.shape
            annos = self.picpac_anno()
            for nodule in annos:
                for j, x, y, rx, ry in nodule:
                    x *= W
                    y *= H
                    rx *= W * aug
                    ry *= H * aug
                    draw = ImageDraw.Draw(frames[j])
                    draw.ellipse([math.floor(x-rx),
                                  math.floor(y-ry),
                                  math.ceil(x+rx),
                                  math.ceil(y+ry)], outline=255)
                    del draw
            pass
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=0.1, loop=0)
        pass

def group_zrange (dcms):
    zs = [float(dcm.dcm.ImagePositionPatient[2]) for dcm in dcms]
    zs = sorted(zs)
    gap = 1000000
    if len(zs) > 1:
        gap = zs[1] - zs[0]
    return (zs[0], zs[-1], gap)

def regroup_dcms (dcms):
    acq_groups = {}
    for dcm in dcms:
        an = 0
        try:
            an = int(dcm.dcm.AcquisitionNumber)
        except:
            pass
        acq_groups.setdefault(an, []).append(dcm)	
        pass
    groups = acq_groups.values()
    if len(groups) == 1:
        return groups[0]
    # we have multiple acquisitions
    zrs = [group_zrange(group) for group in groups]
    zrs = sorted(zrs, key=lambda x: x[0])
    min_gap = min([zr[2] for zr in zrs])
    gap_th = 2.0 * min_gap
    prev = zrs[0]
    bad = False
    for zr in zrs[1:]:
        gap = zr[0] - prev[1]
        if gap < 0 or gap > gap_th:
            bad = True
            break
        if gap != min_gap:
            logging.error('bad gap')
        prev = zr
    if not bad:
        logging.error('multiple acquisitions merged')
        return dcms
    # return the maximal groups
    gs = max([len(group) for group in groups])
    acq_groups = {k:v for k, v in acq_groups.iteritems() if len(v) == gs}
    key = max(acq_groups.keys())
    group = acq_groups[key]
    print(acq_groups.keys(), key)
    logging.error('found conflicting groups. keeping max acq number, %d out of %d dcms' % (len(group), len(dcms)))
    return group

# All DiCOMs of a UID, organized
class FsCase (CaseBase):
    def __init__ (self, path, regroup = True):
        CaseBase.__init__(self)
        self.path = path
        #self.thumb_path = os.path.join(DATA_DIR, 'thumb', uid)
        # load path
        dcms = []
        for dcm_path in glob(os.path.join(self.path, '*.dcm')):
            dcm = dicom.read_file(dcm_path)
            try:
                boxed = DICOM(dcm)
            except:
                print dcm.filename
                raise
            dcms.append(boxed)
            assert dcms[0].spacing == boxed.spacing
            assert dcms[0].shape == boxed.shape
            assert dcms[0].ori_row == boxed.ori_row
            assert dcms[0].ori_col == boxed.ori_col
            if dcms[0].pixel_padding != boxed.pixel_padding:
                logging.warn('0 padding %s, but now %s, %s' %
                        (dcms[0].pixel_padding, boxed.pixel_padding, dcm.filename))
            #assert dcms[0].HU == boxed.HU
            #print boxed.HU
            pass
        assert len(dcms) >= 2

        if regroup:
            dcms = regroup_dcms(dcms)
        self.pixel_padding = dcms[0].pixel_padding

        dcms.sort(key=lambda x: x.position[2])
        zs = []
        for i in range(1, len(dcms)):
            zs.append(dcms[i].position[2] - dcms[i-1].position[2])
            pass
        zs = np.array(zs)
        z_spacing = np.mean(zs)
        assert z_spacing > 0
        assert np.max(np.abs(zs - z_spacing)) * 1000 < z_spacing

        #self.length = dcms[-1].position[2] - dcms[0].position[2]
        front = dcms[0]
        #self.sizes = (front.shape[0] * front.spacing, front.shape[1] * front.spacing, self.length)
        self.dcms = dcms

        images = np.zeros((len(dcms),)+front.image.shape, dtype=np.float32)
        HU = front.HU
        for i in range(len(dcms)):
            HU2 = dcms[i].HU
            images[i,:,:] = dcms[i].image
            if HU2 != HU:
                logging.warn("HU: (%d) %s => %s, %s" % (i, HU2, HU, dcms[i].dcm.filename))
                images[i, :, :] *= HU2[0] / HU[0]
                images[i, :, :] += (HU2[1] - HU[1])/HU[0]
        self.dcm_z_position = {}
        for dcm in dcms:
            name = os.path.splitext(os.path.basename(dcm.dcm.filename))[0]
            self.dcm_z_position[name] = dcm.position[2] - front.position[2]
            pass
        # spacing   # z, y, x
        self.images = images
        self.spacing = (z_spacing, front.spacing, front.spacing)
        x, y, z = front.position
        self.origin = [z, y, z] #front.location
        self.view = AXIAL
        self.anno = None
        self.HU = HU
        self.orig_origin = copy.deepcopy(self.origin)
        self.orig_spacing = copy.deepcopy(self.spacing)
        self.orig_shape = copy.deepcopy(self.images.shape)
        # sanity check
        pass
    pass

class Case:
    def __init__ (self, uid, regroup = True):
        self.uid = uid
        self.path = os.path.join(DATA_DIR, 'bowl', uid)
        if not os.path.exists(self.path):
            self.path = os.path.join(DATA_DIR, 'samples', uid)
            if not os.path.exists(self.path):
                cc = glob(os.path.join(DATA_DIR, 'lymph', 'data', uid, '*/*'))
                if len(cc) >= 1:
                    self.path = cc[0]
                    assert os.path.exists(self.path)
                    if len(cc) > 1:
                        logging.warn('multiple candidates for ' + uid)
                else:
                    cc = glob(os.path.join(DATA_DIR, 'lymph', 'data', '*/*', uid))
                    if len(cc) >= 1:
                        self.path = cc[0]
                        assert os.path.exists(self.path)
                        if len(cc) > 1:
                            logging.warn('multiple candidates for ' + uid)
                    else:
                        raise Exception('data not found for uid %s' % uid)
            pass
        FsCase.__init__(self, self.path, regroup)
        pass
    pass

LUNA_DIR = os.path.join(ROOT, 'data', 'luna')
#LUNA_DIR = os.path.join('data', 'luna')

def load_luna_dir_layout ():
    lookup = {}
    for i in range(10):
        sub = os.path.join(LUNA_DIR, 'subset%d' % i)
        for f in glob(os.path.join(sub, '*.mhd')):
            bn = os.path.splitext(os.path.basename(f))[0]
            #print bn, "=>", sub
            lookup[bn] = sub
            pass
        pass
    return lookup


def load_luna_csv (filename):
    lines = []
    with open(filename, "rb") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
            return lines
        pass
    pass

def load_luna_annotations ():
    ALL = {}
    with open(os.path.join(LUNA_DIR, 'CSVFILES', 'annotations.csv'), 'r') as f:
        f.next()
        for l in f:
            #print l
            uid, x, y, z, d = l.strip().split(',')
            x = float(x)
            y = float(y)
            z = float(z)
            r = float(d)/2
            ALL.setdefault(uid, []).append((z, y, x, r))
            pass
        pass
    return ALL

def load_luna_meta ():
    cache_path = os.path.join(LUNA_DIR, 'meta.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    logging.warn('loading luna meta data')
    meta = (load_luna_dir_layout(),
            load_luna_csv(os.path.join(LUNA_DIR, 'CSVFILES', 'candidates.csv')),
            load_luna_annotations())
    with open(cache_path, 'wb') as f:
        pickle.dump(meta, f)
    return meta

#LUNA_DIR_LOOKUP, _, LUNA_ANNO = load_luna_meta()
LUNA_DIR_LOOKUP = {}
LUNA_ANNO = {}

def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord



# All DiCOMs of a UID, organized
class LunaCase (CaseBase):
    def __init__ (self, uid):
        CaseBase.__init__(self)
        self.uid = uid
        self.path = os.path.join(LUNA_DIR_LOOKUP[uid], uid + '.mhd')
        if not os.path.exists(self.path):
            raise Exception('data not found for uid %s at %s' % (uid, self.path))
        pass
        #self.thumb_path = os.path.join(DATA_DIR, 'thumb', uid)
        # load path
        itkimage = itk.ReadImage(self.path)
        self.HU = (1.0, 0.0)
        self.images = itk.GetArrayFromImage(itkimage).astype(np.float32)
        #print type(self.images), self.images.dtype
        self.origin = list(reversed(itkimage.GetOrigin()))
        self.spacing = list(reversed(itkimage.GetSpacing()))
        self.view = AXIAL
        _, a, b = self.spacing
        self.anno = LUNA_ANNO.get(uid, None)
        assert a == b
        # sanity check
        pass
    pass

def save_mask (path, mask):
    shape = np.array(list(mask.shape), dtype=np.uint32)
    total = mask.size
    totalx = (total +7 )/ 8 * 8
    if totalx == total:
        padded = mask
    else:
        padded = np.zeros((totalx,), dtype=np.uint8)
        padded[:total] = np.reshape(mask, (total,))
        pass
    padded = np.reshape(padded, (totalx/8, 8))
    print padded.shape
    packed = np.packbits(padded)
    print packed.shape
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

def is_kaggle (uid):
    return len(uid) == 32
    
def load_case (uid):
    if is_kaggle(uid):
        return Case(uid)
    else:
        return LunaCase(uid)
    pass


def load_8bit_lungs (uid):
    #path = os.path.join('data/cache', uid)
    #if os.path.exists(path):
    #    with open(path, 'rb') as f:
    #        return pickle.load(f)
    case = load_case(uid)

    case.standardize_color()
    cache = os.path.join('maskcache/mask-123.85-0.01/%s.npz' % case.uid)
    binary = None
    if os.path.exists(cache) and os.path.getsize(cache) > 0:
        # load cache
        binary = load_mask(cache)
        assert not binary is None
    if binary is None:
        binary = segment_lung_axial(case.images) #, th=200.85)
        save_mask(cache, binary)
        pass
    case.images[binary==0] = 255
    case.images *= -1
    case.images += 255
    #case = case.rescale3D(1.0)
    #with open(path, 'wb') as f:
    #    pickle.dump(case, f)
    #return case
    return case

def load_8bit_lungs_noseg (uid):
    #path = os.path.join('data/cache', uid)
    #if os.path.exists(path):
    #    with open(path, 'rb') as f:
    #        return pickle.load(f)
    case = load_case(uid)
    case.standardize_color()
    #case.images = segment_lung_axial(case.images) #, th=200.85)
    #case.images *= -1
    #case.images += 255
    #case = case.rescale3D(1.0)
    #with open(path, 'wb') as f:
    #    pickle.dump(case, f)
    #return case
    return case

def load_16bit_lungs_noseg (uid):
    case = load_case(uid)
    case.standardize_color16()
    return case

def segment_lung_axial_v2 (image, th):

    blur = np.copy(image)
    for i in range(blur.shape[0]):
        cv2.blur(blur[i], (5,5), blur[i])

    binary = np.array(blur < th, dtype=np.uint8)
    # 0: body
    # 1: air & lung
    labels = measure.label(binary, background=-1)

    # set air (same cc as corners) -> body
    bg_labels = set()
    for z in [0, -1]:
        for y in [0, -1]:
            for x in [0, -1]:
                bg_labels.add(labels[z, y, x])
    bg_labels = list(bg_labels)
    print(bg_labels)
    if len(bg_labels) > 1:
        logging.warn('bg not connected, detected %d components' % len(bg_labels))
        pass
    for bg_label in bg_labels:
        binary[bg_label == labels] = 0
        pass


    # now binary:
    #   0: non-lung & body tissue in lung
    #   1: lung & holes in body
    for i, sl in enumerate(binary):
        #H, W = sl.shape
        ll = measure.label(sl, background=-1)   # connected components
        # biggest CC should be body
        vv, cc = np.unique(ll, return_counts=True) 
        assert len(vv) > 0
        body_ll = vv[np.argmax(cc)]
        binary[i][ll != body_ll] = 1
        pass

    # set corner again
    labels = measure.label(binary, background=0)
    bg_labels = set([0])
    for z in [0, -1]:
        for y in [0, -1]:
            for x in [0, -1]:
                bg_labels.add(labels[z, y, x])

    val_counts = zip(*np.unique(labels, return_counts=True))
    val_counts = [x for x in val_counts if not x[0] in bg_labels]   # remove background
    val_counts = sorted(val_counts, key=lambda x:-x[1]) # sort by size
    th = val_counts[0][1] /4    # 1/4 size of the larged connected component (must be lung)
    val = [v for v, c in val_counts if c >= th]
    if len(val) >= 3:
        logging.warn('more than 2 lungs parts detected %d' % len(val))
    binary = np.zeros_like(binary, dtype=np.uint8)
    for v in val:
        binary[labels == v] = 1

    H, W = binary[0].shape
    dilate = int(round(math.sqrt(1.0 * H * W) * dilate))
    #print("DILATE: ", dilate)
    kernel = np.ones((dilate, dilate), dtype=np.int32)

    for i in range(binary.shape[0]):
        cv2.dilate(binary[i], kernel, binary[i])

    #image[binary == 0] = 255
    #image = 255 - image
    #image[binary == 0] = 255
    return binary

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


#def load_lung_mask (uid):
#    #path = os.path.join('data/cache', uid)
#    #if os.path.exists(path):
#    #    with open(path, 'rb') as f:
#    #        return pickle.load(f)
#    case = load_case(uid)
#
#    cache = os.path.join('cache/mask-123.85-0.01/%s.npz' % case.uid)
#    binary = None
#    if os.path.exists(cache) and os.path.getsize(cache) > 0:
#        # load cache
#        binary = load_mask(cache)
#        assert not binary is None
#    if binary is None:
#        case.standardize_color()
#        binary = segment_lung_axial(case.images) #, th=200.85)
#        save_mask(cache, binary)
#        pass
#    case.images = binary.astype(np.float32)
#    return case

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
    y -= size/2
    x -= size/2
    wy = size
    wx = size
    print y, ty, wy, x, tx, wx
    y, ty, wy = patch_clip_range(y, ty, wy, Y)
    x, tx, wx = patch_clip_range(x, tx, wx, X)
    # now do overlap
    patch = np.zeros((size, size, 3), dtype=image.dtype)
    print y, ty, wy, x, tx, wx
    patch[ty:(ty+wy),tx:(tx+wx),:] = image[y:(y+wy),x:(x+wx),:]
    return patch

def eval_ref (test_meta):
    test_ref = load_meta('submits/404-0130.submit', gs=False)
    sum_diff = 0
    sum_adiff = 0
    cases = []
    for (uida, a), (uidb, b) in zip(test_ref, test_meta):
        assert uida == uidb
        sum_diff += b - a
        sum_adiff += abs(b -a)
        cases.append((uida, b-a, a, b))
    top = sorted(cases, key=lambda x: -x[1])[:5]
    bot = sorted(cases, key=lambda x: x[1])[:5]
    print "INFLATION:", sum_diff
    print "ABS DIFF:", sum_adiff
    for uid, x, a, b in top + bot:
        print uid, x, a, b
        pass
    pass


if __name__ == '__main__':
    #dump_meta('a', STAGE1.train)
    #dump_meta('b', STAGE1.test)
    pass
