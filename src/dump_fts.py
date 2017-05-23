#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from adsb3 import *

dim, fts = load_fts(sys.argv[1])

print(dim)
for w, pos, one in fts:
    print(w, pos, one)

