#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
from glob import glob
import copy
import cv2
import dicom
import numpy as np
import SimpleITK as itk
import logging
from _plumo import *

# configuration options
DICOM_STRICT = False

def dicom_error (dcm, msg, level=logging.ERROR):
    s = 'DICOM ERROR (%s): %s' % (dcm.filename, msg)
    if DICOM_STRICT and level >= logging.ERROR:
        raise Exception(s)
    else:
        logging.log(level, s)
    pass

class Dicom:
    def __init__ (self, dcm):
        # slope & intercept for rescaling to Hounsfield unit
        self.HU = (float(dcm.RescaleSlope), float(dcm.RescaleIntercept))
        # filename as slice ID
        #self.sid = os.path.splitext(os.path.basename(dcm.filename))[0]
        self.dcm = dcm
        self.image = dcm.pixel_array
        self.shape = dcm.pixel_array.shape
        self.pixel_padding = None
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

AXIAL, SAGITTAL, CORONAL = 0, 1, 2

VIEWS = [AXIAL, SAGITTAL, CORONAL]
VIEW_NAMES = ['axial', 'sagittal', 'coronal']

AXES_ORDERS = ([0, 1, 2],  # AXIAL
               [2, 1, 0],  # SAGITTAL
               [1, 0, 2])  # CORONAL


#def index_view (I, view):
#    assert len(I) == 3
#    a, b, c = AXES_ORDERS[view]
#    return [I[a], I[b], I[c]]

class VolumeBase (object):
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

        self.orig_origin = None
        self.orig_spacing = None
        self.orig_shape = None
        pass

    def copy (self):
        volume = VolumeBase()
        volume.uid = self.uid
        volume.path = self.path
        volume.images = copy.deepcopy(self.images)
        volume.spacing = self.spacing
        volume.origin = self.origin
        volume.view = self.view
        volume.anno = self.anno
        volume.HU = self.HU
        volume.orig_origin = self.orig_origin
        volume.orig_spacing = self.orig_spacing
        volume.orig_shape = self.orig_shape
        return volume

    # return a copy of volume, with images replaced
    def copy_replace_images (self, images):
        volume = self.copy()
        volume.images = images
        return volume

    def normalizeHU (self):
        assert not self.HU is None
        a, b = self.HU
        self.images *= a
        self.images += b
        self.HU = (1.0, 0)
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

        volume = self.copy()
        volume.images = np.swapaxes(self.images, d1, d2)
        assert isinstance(self.spacing, tuple)
        sp = list(self.spacing)
        sp[d1], sp[d2] = sp[d2], sp[d1]
        volume.spacing = tuple(sp)
        volume.view = view
        return volume

    # clip or pad the volume so size along all axies are the give size
    def force_size (self, size=512):

        def clip_or_pad (n):
            if n >= size:
                from_x = (n-size)//2
                to_x = 0
                n_x = size
                shift_x = from_x
            else:
                from_x = 0
                to_x = (size-n)//2
                n_x = n
                shift_x = -to_x
            return from_x, to_x, n_x, shift_x

        target = np.zeros((size,size,size), dtype=self.images.dtype)
        Z, Y, X = self.images.shape
        from_z, to_z, n_z, shift_z = clip_or_pad(Z)
        from_y, to_y, n_y, shift_y = clip_or_pad(Y)
        from_x, to_x, n_x, shift_x = clip_or_pad(X)
        target[to_z:(to_z+n_z),
               to_y:(to_y+n_y),
               to_x:(to_x+n_x)] = self.images[from_z:(from_z+n_z),
                   from_y:(from_y+n_y),
                   from_x:(from_x+n_x)]
        self.origin[0] += shift_z * self.spacing[0]
        self.origin[1] += shift_y * self.spacing[1]
        self.origin[2] += shift_x * self.spacing[2]
        #print("off", to_x, to_y, to_z)
        #print("len", n_x, n_y, n_z)
        #print("shi", shift_x, shift_y, shift_z)
        self.images = target
        pass

    # round the size down so each axis is divisible by stride
    def trunc_size (self, stride=16):
        T, H, W = self.images.shape[:3]
        nT = T // stride * stride
        nH = H // stride * stride
        nW = W // stride * stride
        oT = (T - nT)//2
        oH = (H - nH)//2
        oW = (W - nW)//2
        self.origin[0] += oT * self.spacing[0]
        self.origin[1] += oH * self.spacing[1]
        self.origin[2] += oW * self.spacing[2]
        self.images = self.images[oT:(oT+nT),oH:(oH+nH),oW:(oW+nW)]
        pass

    # trim array by mask, removing edges whose value in mask is 0
    def trim (self, mask, margin1=2, margin2=10):
        def strip_loc (array1d, margin=0):
            w = np.where(array1d > 0)
            x0 = np.min(w)
            x1 = np.max(w)+1
            return max(0, x0-margin), min(x1+margin, array1d.shape[0])
        z0, z1 = trim_loc(np.sum(mask, axis=(1,2)), margin=margin1)
        y0, y1 = trim_loc(np.sum(mask, axis=(0,2)), margin=margin2)
        x0, x1 = trim_loc(np.sum(mask, axis=(0,1)), margin=margin2)
        self.origin[0] += z0 * self.spacing[0]
        self.origin[1] += y0 * self.spacing[1]
        self.origin[2] += x0 * self.spacing[2]
        self.images = self.images[z0:z1, y0:y1, x0:x1]
        pass

    # consider using scipy.ndimage.interpolation
    def rescale_helper (self, slices = None, spacing = None, size = None, method=2):
        # if slices != self.images.shape[0], use method:
        #   0:  adjust slices, so everything is integer and no rounding or approx. is done
        #   1:  do not change slices, use nearest neighbor
        #   2:  do not change slices, use interpolation
        N, H, W = self.images.shape
        volume = self.copy()
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
            slices = N // step
            off = (N - slices * step) // 2
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

        volume.spacing = (sp1, sp2, sp3)
        volume.images = np.zeros((slices, H, W), dtype=np.float32)

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
                cv2.resize(image, resize, volume.images[i, :, :])
            else:
                volume.images[i, :, :] = image
            off += step
        return volume

    # rescale images so spacing along each axis is the give value
    def rescale_spacing (self, spacing):
        slices = int(round(self.spacing[0] * (self.images.shape[0] - 1) / spacing + 1))
        return self.rescale_helper(slices, spacing, size=None, method=2)
    pass

    # clip color values to [min_th, max_th] and then normalize to [min, max]
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

    def normalize_8bit (self):
        self.normalizeHU()
        self.normalize(min_th=-1000,max_th=400,min=0,max=255)
        pass

    def normalize_16bit (self):
        self.normalizeHU()
        self.normalize(min_th=-1000,max_th=400,min=0,max=1400)
        pass
    pass


# The dicom files of a volume might have been merged from multiple image
# acquisitions and might not form a coherent volume
# try to merge multiple acquisitions.  If that's not possible, keep the one
# with most slices
def regroup_dicom_slices (dcms):

    def group_zrange (dcms):
        zs = [float(dcm.dcm.ImagePositionPatient[2]) for dcm in dcms]
        zs = sorted(zs)
        gap = 1000000
        if len(zs) > 1:
            gap = zs[1] - zs[0]
        return (zs[0], zs[-1], gap)

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
class DicomVolume (VolumeBase):
    def __init__ (self, path, uid=None, regroup = True):
        VolumeBase.__init__(self)
        self.uid = uid
        self.path = path
        #self.thumb_path = os.path.join(DATA_DIR, 'thumb', uid)
        # load path
        dcms = []
        for dcm_path in glob(os.path.join(self.path, '*.dcm')):
            dcm = dicom.read_file(dcm_path)
            try:
                boxed = Dicom(dcm)
            except:
                print(dcm.filename)
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
            dcms = regroup_dicom_slices(dcms)

        dcms.sort(key=lambda x: x.position[2])
        zs = []
        for i in range(1, len(dcms)):
            zs.append(dcms[i].position[2] - dcms[i-1].position[2])
            pass
        zs = np.array(zs)
        z_spacing = np.mean(zs)
        assert z_spacing > 0
        assert np.max(np.abs(zs - z_spacing)) * 100 < z_spacing

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
        #self.dcm_z_position = {}
        #for dcm in dcms:
        #    name = os.path.splitext(os.path.basename(dcm.dcm.filename))[0]
        #    self.dcm_z_position[name] = dcm.position[2] - front.position[2]
        #    pass
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

# MHD volume
class MetaImageVolume (VolumeBase):
    def __init__ (self, path, uid=None):
        VolumeBase.__init__(self)
        self.uid = uid
        self.path = path
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
        #self.anno = LUNA_ANNO.get(uid, None)
        assert a == b
        # sanity check
        pass
    pass

def default_pyramid_extractor (dim, nodules):
    if len(nodules) == 0:
        return [0] * dim
    else:
        return nodules[0][2][:dim]

class Pyramid:
    def __init__ (self, partitions = [(1,1,1)]):
        self.partitions = partitions
        parts = 0
        for x, y, z in self.partitions:
            parts += x * y * z
            pass
        self.total = parts
        pass

    def parts (self):
        return self.total

    def apply (self, dim, nodules, extractor=default_pyramid_extractor):
        parts = []
        for _ in range(self.total):
            parts.append([])
        for w, pos, ft in nodules:
            z, y, x = pos
            off = 0
            for LZ, LY, LX in self.partitions:
                zi = min(int(math.floor(z * LZ)), LZ-1)
                yi = min(int(math.floor(y * LY)), LY-1)
                xi = min(int(math.floor(x * LX)), LX-1)
                pi = off + (zi * LY + yi) * LX + xi
                off += LZ * LY * LX
                assert pi < off
                parts[pi].append((w, pos, ft))
                pass
            #assert off == TOTAL_PARTS
            pass
        ft = []
        for nodules in parts:
            ft.extend(extractor(dim, nodules))
            pass
        return ft

