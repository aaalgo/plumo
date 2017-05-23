#!/usr/bin/env python
import sys
import math
import time
import traceback
import subprocess
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import meta_graph
from scipy.ndimage.morphology import grey_dilation, binary_dilation
from skimage import measure
import plumo
from adsb3 import *

BATCH = 32

def setGpuConfig (config):
    mem_total = subprocess.check_output('nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits', shell=True)
    mem_total = float(mem_total)
    frac = 5000.0/mem_total
    print("setting GPU memory usage to %f" % frac)
    if frac < 0.5:
        config.gpu_options.per_process_gpu_memory_fraction = frac
    pass

def extract (prob, fts, th=0.05, ext=2):
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
        dim = fts.shape[3]

    Z, Y, X = prob.shape
    for box in boxes:
        #print prob.shape, fts.shape
        z0, y0, x0, z1, y1, x1 = box.bbox
        #ft.append((z1-z0)*(y1-y0)*(x1-x0))
        prob_roi = prob[z0:z1,y0:y1,x0:x1]
        za, ya, xa, zz, zy, zx, yy, yx, xx = plumo.norm3d(prob_roi)
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
        #print eig
        #print weight_sum, np.linalg.det(cov), eig
        #print za, ya, xa
        #print cov

        pos = (zc/Z, yc/Y, xc/X)

        if fts is None:
            one = [prob_sum, math.atan2(eig[0], eig[2])]
        else:
            fts_roi = fts[z0:z1,y0:y1,x0:x1,:]
            fts_sum = np.sum(fts_roi, axis=(0,1,2))
            one = list(fts_sum/weight_sum)
        nodules.append((prob_sum, pos, one))
        pass
    nodules = sorted(nodules, key=lambda x: -x[0])
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
        print dir_path
        paths = glob(os.path.join(dir_path, '*.meta'))
        print paths
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

        pp = '%d' % FLAGS.bits
        if SPACING != 0.8:
            pp += '_%.1f' % SPACING
        if GAP != 5:
            pp += '_%d' % GAP
        print "PP:", pp

        models = []
        if fts_model is None:
            models.append(None)
        else:
            models.append(ViewModel(X4, FTS_KEEP, plumo.AXIAL, 'fts', 'models/%s' % fts_model, node='fts:0', softmax=False))

        if prob_mode == MODE_AXIAL:
            models.append(ViewModel(X4, PROB_KEEP, plumo.AXIAL, 'axial', 'models/%s/axial' % prob_model))
        elif prob_mode == MODE_SAGITTAL:
            models.append(ViewModel(X4, PROB_KEEP, plumo.SAGITTAL, 'sagittal', 'models/%s/sagittal' % prob_model))
        elif prob_mode == MODE_CORONAL:
            models.append(ViewModel(X4, PROB_KEEP, plumo.CORONAL, 'coronal', 'models/%s/coronal' % prob_model))
        else:
            models.append(ViewModel(X4, PROB_KEEP, plumo.AXIAL, 'axial', 'models/%s/axial' % prob_model))
            models.append(ViewModel(X4, PROB_KEEP, plumo.SAGITTAL, 'sagittal', 'models/%s/sagittal' % prob_model))
            models.append(ViewModel(X4, PROB_KEEP, plumo.CORONAL, 'coronal', 'models/%s/coronal' % prob_model))
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
        views = [case.transpose(plumo.AXIAL)]
        if self.mode > MODE_AXIAL:
             views.append(case.transpose(plumo.SAGITTAL))
             views.append(case.transpose(plumo.CORONAL))
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
            if m.view != plumo.AXIAL:
                fts = cc.transpose_array(plumo.AXIAL, fts)
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

        if mask:
            pre_sum = np.sum(prob)
            prob *= mask.images
            post_sum = np.sum(prob)
            logging.info('mask reduction %f' % ((pre_sum-post_sum)/pre_sum))
        prob = np.ascontiguousarray(prob)
        return extract(prob, r[0])
    pass


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('prob', 'luna.ns.3c', 'prob model')      # prob model
#original default is luna.ns.3c
flags.DEFINE_string('fts', None, 'fts model')                # ft model
flags.DEFINE_string('mask', None, 'mask')
flags.DEFINE_integer('mode', MODE_AXIAL, '')              # use axial instead of min of 3 views
flags.DEFINE_integer('channels', 3, '')
flags.DEFINE_integer('bits', 16, '')
flags.DEFINE_integer('stride', 16, '')
flags.DEFINE_integer('dilate', 10, '')
flags.DEFINE_bool('prob_dropout', False, '')
flags.DEFINE_bool('fts_dropout', False, '')
flags.DEFINE_bool('fast', False, '')


def main (argv):
    model = Model(FLAGS.prob, FLAGS.mode, FLAGS.fts, FLAGS.channels, FLAGS.prob_dropout, FLAGS.fts_dropout)
    name = FLAGS.prob
    if FLAGS.fts:
        name += '_' + FLAGS.fts
    if FLAGS.mode != MODE_AXIAL:
        name += '_m' + str(FLAGS.mode)
    if FLAGS.channels != 3:
        name += '_c' + str(FLAGS.channels)
    if FLAGS.bits != 16:
        name += '_b' + str(FLAGS.bits)
    if not FLAGS.mask is None:
        name += '_' + FLAGS.mask
    if FLAGS.dilate != 10:
        name += '_d' + str(FLAGS.dilate)
    if SPACING != 0.8:
        name += '_s%.1f' % SPACING
    if GAP != 5:
        name += '_g%d' % GAP

    #name = '%s_%s_%d_%d' % (FLAGS.prob, FLAGS.fts, FLAGS.mode, FLAGS.channels)
    ROOT = os.path.join('cache', name)
    try_mkdir(ROOT)

    config = tf.ConfigProto()
    setGpuConfig(config)
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        model.load(sess)
        for uid, _ in ALL_CASES:
            cache = os.path.join(ROOT, uid + '.pkl')
            if os.path.exists(cache):
                continue
            with open(cache, 'wb') as f:
                pass
            start_time = time.time()
            if FLAGS.bits == 8:
                case = load_8bit_lungs_noseg(uid)
            elif FLAGS.bits == 16:
                case = load_16bit_lungs_noseg(uid)
            else:
                assert False
            load_time = time.time()
            mask = None
            if not FLAGS.mask is None:
                try:
                    mask_path = 'cache/%s/%s.npz' % (FLAGS.mask, uid)
                    mask = load_mask(mask_path)
                    mask = case.copy_replace_images(mask.astype(dtype=np.float32))
                    mask = mask.rescale_spacing(SPACING)
                    mask.trunc_size(FLAGS.stride)
                    if FLAGS.dilate > 0:
                        #print 'dilate', FLAGS.dilate
                        ksize = FLAGS.dilate * 2 + 1
                        mask.images = grey_dilation(mask.images, size=(ksize, ksize, ksize), mode = 'constant')
                except:
                    traceback.print_exc()
                    logging.error('failed to load mask %s' % mask_path)
                    mask = None

            case = case.rescale_spacing(SPACING)
            case.trunc_size(FLAGS.stride)
            dim, nodules = model.apply(sess, case, mask)
            predict_time = time.time()
            with open(cache, 'wb') as f:
                pickle.dump((dim, nodules), f)
                pass
            print uid, (load_time - start_time), (predict_time - load_time)
        pass

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()

