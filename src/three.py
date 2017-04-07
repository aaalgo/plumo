#!/usr/bin/env python
import os
import sys
import shutil
import subprocess
from glob import glob
from jinja2 import Environment, FileSystemLoader
from adsb3 import DATA_DIR
import pyadsb3

#TMPL_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),
#                './templates')
STATIC_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                './static')
#env = Environment(loader=FileSystemLoader(searchpath=TMPL_DIR))
#case_tmpl = env.get_template('three_case.html')
#$index_tmpl = env.get_template('papaya_index.html')

def Three (path, verts, faces):
    try:
        os.makedirs(path)
    except:
        pass
    try:
        subprocess.check_call('rm -rf %s/*' % path, shell=True)
    except:
        pass
    pyadsb3.save_mesh(verts, faces, os.path.join(path, 'model.ply'))
        #try:
        #    data = os.path.abspath(DATA_DIR)
        #    os.symlink(data, os.path.join(path, 'data'))
        #except:
        #    pass
    for f in ['three.js', 'PLYLoader.js', 'Detector.js', 'TransformControls.js']:
        shutil.copyfile(os.path.join(STATIC_DIR, f), os.path.join(path, f))
    shutil.copyfile(os.path.join(STATIC_DIR, 'three_index.html'), os.path.join(path, 'index.html'))
    pass

