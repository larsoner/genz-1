#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

import os
import os.path as op
import sys

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G = nx.from_numpy_array(arr)
plt.subplot(121)
nx.draw(G, with_labels=True)