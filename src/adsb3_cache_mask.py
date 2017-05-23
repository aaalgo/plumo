#!/usr/bin/env python
import sys
import time
import subprocess
import numpy as np
import mesh
from adsb3 import *

try_mkdir('cache/mask')
try_mkdir('cache/hull')
for uid, _ in ALL_CASES[:1]:
    cache = os.path.join('cache/mask', uid + '.npz')
    cacheh = os.path.join('cache/hull', uid + '.npz')
    if os.path.exists(cache) and os.path.exists(cacheh):
        continue
    with open(cache, 'wb') as f:
        pass
    start_time = time.time()
    case = Case(uid)
    case.normalizeHU()
    spacing = case.spacing
    UNIT = spacing[0] * spacing[1] * spacing[2]
    binary, body_counts = mesh.segment_lung(case.images) #, smooth=20)
    ft = (1, [(x * UNIT, [x]) for x in body_counts])
    save_mask(cache, binary)
    binary = mesh.convex_hull(binary)
    save_mask(cacheh, binary)
    load_time = time.time()
    print uid, (load_time - start_time)
pass

