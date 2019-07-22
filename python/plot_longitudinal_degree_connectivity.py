#!/usr/bin/env python

"""plot_longitudinal_degree_connectivity.py: plot spaghetti plots of
functional connectivity index for ROIs."""

__author__ = 'Kambiz Tavabi'
__copyright__ = 'Copyright 2019, Seattle, Washington'
__credits__ = []
__license__ = 'MIT'
__version__ = '1.0.1'
__maintainer__ = 'Kambiz Tavabi'
__email__ = 'ktavabi@uw.edu'
__status__ = 'Production'

import os.path as op
import pandas as pd
import seaborn as sns
from python import defaults

sns.set(style='whitegrid', color_codes=True)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.set_option('precision', 2)

age = [9]
bands = [('14-30')]
deg_df = pd.read_csv(op.join(defaults.datadir,
                             'genz_%dyrs_%sHz_deg-conn.csv' % (aix, band)))
f, ax = plt.subplots(figsize=(7, 18))
    sns.boxplot(x='deg', y='roi', data=df.dropna(),
                whis='range',
                hue='hem', palette=['m', 'g'])
    ax.set_title('%syrs_%d - %dHz degrees' %
                 (age, bands[0][0], bands[0][1]))
    f.tight_layout()
