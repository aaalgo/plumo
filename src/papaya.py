#!/usr/bin/env python
import os
import sys
import shutil
import subprocess
from glob import glob
from jinja2 import Environment, FileSystemLoader
from adsb3 import DATA_DIR

TMPL_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                './templates')
STATIC_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                './static')
env = Environment(loader=FileSystemLoader(searchpath=TMPL_DIR))
case_tmpl = env.get_template('papaya_case.html')
index_tmpl = env.get_template('papaya_index.html')

class Annotations:
    def __init__ (self):
        self.annos = []
        pass

    def add (self, box, hint=None):
        self.annos.append({'box': box, 'hint': hint})
        pass

def Papaya (path, case, annotations=Annotations(), images = None, text = ''):
    try:
        os.makedirs(path)
    except:
        pass
    try:
        subprocess.check_call('rm -rf %s/dcm' % path, shell=True)
    except:
        pass
        #try:
        #    data = os.path.abspath(DATA_DIR)
        #    os.symlink(data, os.path.join(path, 'data'))
        #except:
        #    pass
    for f in ['papaya.css', 'papaya.js']:
        shutil.copyfile(os.path.join(STATIC_DIR, f), os.path.join(path, f))
        pass
    pass
    os.mkdir(os.path.join(path, 'dcm'))
    subprocess.check_call('cp %s/*.dcm %s/dcm/' % (case.path, path), shell=True)
    images = glob(os.path.join(path, 'dcm/*.dcm'))
    images = ['/'.join(x.split('/')[-2:]) for x in images]
    boxes = []
    centers = []
    for anno in annotations.annos:
        box = case.papaya_box(anno['box'])
        boxes.append(box)
        z1, y1, x1, z2, y2, x2 = box
        hint = anno.get('hint', None)
        center = ((z1+z2)/2, (y1+y2)/2, (x1+x2)/2,hint)
        centers.append(center)
    with open(os.path.join(path, 'index.html'), 'w') as f:
        f.write(case_tmpl.render(images=images, boxes=boxes, centers=centers))
        pass
    pass

if __name__ == '__main__':
    from adsb3 import Case
    case = Case('008464bb8521d09a42985dd8add3d0d2')
    papaya = Papaya('/home/wdong/public_html/papaya_test')
    boxes =[[38, 359, 393, 42, 367, 404], [63, 189, 138, 64, 201, 156], [64, 208, 82, 66, 218, 90], [126, 227, 343, 128, 237, 351], [138, 385, 180, 139, 391, 186]]
    papaya.next(case, boxes=boxes)
    papaya.flush()
