#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import argparse
import math
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import meta_graph
from scipy.ndimage.morphology import grey_dilation, binary_dilation
from scipy.ndimage.filters import gaussian_filter
from skimage import measure
from adsb3 import *
import pyadsb3
import mesh
from papaya import Papaya, Annotations
from three import Three

BATCH = 32
MIN_NODULE_SIZE=30
PARTITIONS = [(1,1,1),(1,1,2),(3,1,1),(1,1,3)]
TOTAL_PARTS = 0
for x, y, z in PARTITIONS:
    TOTAL_PARTS += x * y * z
    pass

def extract_nodules (prob, fts, th=0.05, ext=2):
    if not fts is None:
        prob4 = np.reshape(prob, prob.shape + (1,))
        assert prob4.base is prob
        fts = np.clip(fts, 0, 6)
        fts *= prob4
    binary = prob > th
    k = int(round(ext / SPACING))
    binary = binary_dilation(binary, iterations=k)
    labels = measure.label(binary, background=0)
    boxes = measure.regionprops(labels)

    nodules = []
    dim = 2
    if not fts is None:
        dim += fts.shape[3]

    Z, Y, X = prob.shape
    Z = 1.0 * Z
    Y = 1.0 * Y
    X = 1.0 * X
    for box in boxes:
        #print prob.shape, fts.shape
        z0, y0, x0, z1, y1, x1 = box.bbox
        #ft.append((z1-z0)*(y1-y0)*(x1-x0))
        prob_roi = prob[z0:z1,y0:y1,x0:x1]
        za, ya, xa, zz, zy, zx, yy, yx, xx = pyadsb3.norm3d(prob_roi)
        zc = za + z0
        yc = ya + y0
        xc = xa + x0

        cov = np.array([[zz, zy, zx],
                        [zy, yy, yx],
                        [zx, yx, xx]], dtype=np.float32)
        eig, _ = np.linalg.eig(cov)
        #print zc, yc, xc, '------', (z0+z1)/2.0, (y0+y1)/2.0, (x0+x1)/2.0

        weight_sum = np.sum(prob_roi)
        UNIT = SPACING * SPACING * SPACING
        prob_sum = weight_sum * UNIT

        eig = sorted(list(eig), reverse=True)

        pos = (zc/Z, yc/Y, xc/X)
        #box = (z0/Z, y0/Y, x0/X, z1/Z, y1/Y, x1/X)

        one = [prob_sum, math.atan2(eig[0], eig[2])]
        if not fts is None:
            fts_roi = fts[z0:z1,y0:y1,x0:x1,:]
            fts_sum = np.sum(fts_roi, axis=(0,1,2))
            one.extend(list(fts_sum/weight_sum))
        nodules.append((prob_sum, pos, one, box.bbox))
        pass
    return dim, nodules

def logits2prob (v, scope='logits2prob'):
    with tf.name_scope(scope):
        shape = tf.shape(v)    # (?, ?, ?, 2)
        # softmax
        v = tf.reshape(v, (-1, 2))
        v = tf.nn.softmax(v)
        v = tf.reshape(v, shape)
        # keep prob of 1 only
        v = tf.slice(v, [0, 0, 0, 1], [-1, -1, -1, -1])
        # remove trailing dimension of 1
        v = tf.squeeze(v, axis=3)
    return v

class ViewModel:
    def __init__ (self, X, KEEP, view, name, dir_path, node='logits:0', softmax=True):
        self.name = name
        self.view = view
        paths = glob(os.path.join(dir_path, '*.meta'))
        assert len(paths) == 1
        path = os.path.splitext(paths[0])[0]
        mg = meta_graph.read_meta_graph_file(path + '.meta')
        if KEEP is None:
            fts, = tf.import_graph_def(mg.graph_def, name=name,
                    input_map={'images:0':X},
                                return_elements=[node])
        else:
            fts, = tf.import_graph_def(mg.graph_def, name=name,
                    input_map={'images:0':X, 'keep:0':KEEP},
                                return_elements=[node])
        if softmax:
            fts = logits2prob(fts)
        self.fts = fts
        self.saver = tf.train.Saver(saver_def=mg.saver_def, name=name)
        self.loader = lambda sess: self.saver.restore(sess, path)
        pass

MODE_AXIAL = 1
MODE_SAGITTAL = 2
MODE_CORONAL = 3
MODE_MIN   = 4

class Model:
    def __init__ (self, prob_model, prob_mode, fts_model, channels = 3, prob_dropout=False, fts_dropout=True):
        if channels == 1:
            self.X = tf.placeholder(tf.float32, shape=(None, None, None))
            X4 = tf.expand_dims(self.X, axis=3)
        elif channels == 3:
            self.X = tf.placeholder(tf.float32, shape=(None, None, None, channels))
            X4 = self.X 
        else:
            assert False
        self.KEEP = tf.placeholder(tf.float32, shape=())
        PROB_KEEP = None
        FTS_KEEP = None
        if prob_dropout:
            PROB_KEEP = self.KEEP
        if fts_dropout:
            FTS_KEEP = self.KEEP

        models = []
        if fts_model is None:
            models.append(None)
        else:
            models.append(ViewModel(X4, FTS_KEEP, AXIAL, 'fts', 'models/%s' % fts_model, node='fts:0', softmax=False))

        if prob_mode == MODE_AXIAL:
            models.append(ViewModel(X4, PROB_KEEP, AXIAL, 'axial', 'models/%s/axial' % prob_model))
        elif prob_mode == MODE_SAGITTAL:
            models.append(ViewModel(X4, PROB_KEEP, SAGITTAL, 'sagittal', 'models/%s/sagittal' % prob_model))
        elif prob_mode == MODE_CORONAL:
            models.append(ViewModel(X4, PROB_KEEP, CORONAL, 'coronal', 'models/%s/coronal' % prob_model))
        else:
            models.append(ViewModel(X4, PROB_KEEP, AXIAL, 'axial', 'models/%s/axial' % prob_model))
            models.append(ViewModel(X4, PROB_KEEP, SAGITTAL, 'sagittal', 'models/%s/sagittal' % prob_model))
            models.append(ViewModel(X4, PROB_KEEP, CORONAL, 'coronal', 'models/%s/coronal' % prob_model))
        self.channels = channels
        self.models = models
        self.mode = prob_mode
        pass

    def load (self, sess):
        for m in self.models:
            if m:
                m.loader(sess)
        pass

    def apply (self, sess, case, mask):
        r = []
        #comb = np.ones_like(case.images, dtype=np.float32)
        views = [case.transpose(AXIAL)]
        if self.mode > MODE_AXIAL:
             views.append(case.transpose(SAGITTAL))
             views.append(case.transpose(CORONAL))
        for m in self.models:
            if m is None:
                r.append(None)
                continue
            cc = views[m.view]
            images = cc.images
            N, H, W = images.shape

            fts = None #np.zeros_like(images, dtype=np.float32)
            margin = 0
            if self.channels == 3:
                margin = GAP

            fts = None
            off = margin
            while off < N-margin:
                nxt = min(off + BATCH, N-margin)
                x = np.zeros((nxt-off, H, W, FLAGS.channels), dtype=np.float32)
                i = 0
                for j in range(off, nxt):
                    if self.channels == 1:
                        x[i] = images[j]
                    elif self.channels == 3:
                        x[i,:,:,0] = images[j-GAP]
                        x[i,:,:,1] = images[j]
                        x[i,:,:,2] = images[j+GAP]
                    else:
                        assert False
                    i += 1
                    pass
                assert i == x.shape[0]
                y, = sess.run([m.fts], feed_dict={self.X:x, self.KEEP:1.0})
                if fts is None:
                    fts = np.zeros((N,) + y.shape[1:], dtype=np.float32)
                fts[off:nxt] = y
                off = nxt
                pass
            assert off == N - margin
            if m.view != AXIAL:
                fts = cc.transpose_array(AXIAL, fts)
            r.append(fts)
            pass
        if len(r) == 2:
            prob = r[1]
        elif len(r) == 4:
            prob = r[1]
            np.minimum(prob, r[2], prob)
            np.minimum(prob, r[3], prob)
        else:
            assert False

        if not mask is None:
            pre_sum = np.sum(prob)
            prob *= mask
            post_sum = np.sum(prob)
            logging.info('mask reduction %f' % ((pre_sum-post_sum)/pre_sum))
        prob = np.ascontiguousarray(prob)
        return extract_nodules(prob, r[0])
    pass


def combine (dim, nodules):
    if len(nodules) == 0 or nodules[0][0] < MIN_NODULE_SIZE:
        return [0] * dim
    else:
        return nodules[0][2]

def pyramid (dim, nodules):
    parts = []
    for _ in range(TOTAL_PARTS):
        parts.append([])
    for w, pos, ft, box in nodules:
        z, y, x = pos
        off = 0
        for LZ, LY, LX in PARTITIONS:
            zi = min(int(math.floor(z * LZ)), LZ-1)
            yi = min(int(math.floor(y * LY)), LY-1)
            xi = min(int(math.floor(x * LX)), LX-1)
            pi = off + (zi * LY + yi) * LX + xi
            off += LZ * LY * LX
            assert pi < off
            parts[pi].append((w, pos, ft))
            pass
        assert off == TOTAL_PARTS
        pass
    ft = []
    for nodules in parts:
        ft.extend(combine(dim, nodules))
        pass
    return ft

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('prob', 'nodule', 'prob model')      # prob model
#original default is luna.ns.3c
flags.DEFINE_string('fts', 'ft', 'fts model')                # ft model
flags.DEFINE_string('score', 'score', 'score model')                # ft model
#flags.DEFINE_string('mask', None, 'mask')
flags.DEFINE_integer('mode', MODE_AXIAL, '')              # use axial instead of min of 3 views
flags.DEFINE_integer('channels', 3, '')
flags.DEFINE_integer('bits', 8, '')
flags.DEFINE_integer('stride', 16, '')
flags.DEFINE_integer('dilate', 10, '')
flags.DEFINE_bool('prob_dropout', False, '')
flags.DEFINE_bool('fts_dropout', True, '')
flags.DEFINE_string('input', None, '')
flags.DEFINE_string('output', None, '')

def pred_wrap (Xin):
    Yout = model.predict_proba(Xin)[:,1]
    return Yout

def save_mesh (binary, path):
    binary = mesh.pad(binary, dtype=np.float) 
    binary = gaussian_filter(binary, 2, mode='constant')
    verts, faces = measure.marching_cubes(binary, 0.5)
    Three(path, verts, faces)

def main (argv):
    nodule_model = Model(FLAGS.prob, FLAGS.mode, FLAGS.fts, FLAGS.channels, FLAGS.prob_dropout, FLAGS.fts_dropout)
    with open(os.path.join('models', FLAGS.score), 'rb') as f:
        score_model = pickle.load(f)

    case = FsCase(FLAGS.input)

    case.normalizeHU()
    case = case.rescale3D(SPACING)
    lung, _ = mesh.segment_lung(case.images)
    save_mesh(lung, os.path.join(FLAGS.output, 'lung'))
    mask = mesh.convex_hull(lung)
    #body, _ = mesh.segment_body(case.images)
    #save_mesh(body, os.path.join(FLAGS.output, 'body'))
    case.standardize_color()

    case.round_stride(FLAGS.stride)

    mask = case.copy_replace_images(mask)
    mask.round_stride(FLAGS.stride)
    mask = mask.images

    if FLAGS.dilate > 0:
        ksize = FLAGS.dilate * 2 + 1
        mask = grey_dilation(mask, size=(ksize, ksize, ksize), mode='constant')
        pass
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        nodule_model.load(sess)
        dim, nodules = nodule_model.apply(sess, case, mask)
        pass
    pass


    fts = []
    pos = []
    for nodule in nodules:
        fts.append(pyramid(dim, [nodule]))
        pos.append(nodule[3])
        pass
    Nt = np.array(fts, dtype=np.float32)
    Ny = score_model.predict_proba(Nt)[:,1]
    pw = sorted(zip(pos, list(Ny)), key=lambda x:x[1], reverse=True)
    anno = Annotations()
    for box, score in pw:
        if score < 0.1:
            break
        anno.add(box, str(score))
        pass
    Papaya(FLAGS.output, case, annotations=anno)
    pass

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()

