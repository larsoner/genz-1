#!/usr/bin/env python

"""plot_COH_connectivity.py: Plot functional connectivity (degree).
    Does per age:
        1.
"""

__author__ = 'Kambiz Tavabi'
__copyright__ = 'Copyright 2018, Seattle, Washington'
__credits__ = ['Eric Larson']
__license__ = 'MIT'
__version__ = '1.0.1'
__maintainer__ = 'Kambiz Tavabi'
__email__ = 'ktavabi@uw.edu'
__status__ = 'Development'

import os.path as op
import re

import mne
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from meeg_preprocessing import utils
from surfer import Brain
import matplotlib.pyplot as plt
from genz import defaults

sns.set_style('ticks')
sns.set_palette('colorblind')
utils.setup_mpl_rcparams(font_size=10)

# define frequencies of interest
bands = {
    'DC': (0.01, 2), 'delta': (2, 4), 'theta': (5, 7),
    'alpha': (8, 12), 'beta': (13, 29), 'gamma': (30, 50)
    }
ages = np.arange(9, 19, 2)
lims = [75, 85, 95]
# Load labels
fslabels = mne.read_labels_from_annot('fsaverage', 'aparc_sub', 'both',
                                      subjects_dir=defaults.subjects_dir)
fslabels = [label for label in fslabels
            if not label.name.startswith('unknown')]
brain = Brain('fsaverage', 'lh', 'inflated', subjects_dir=defaults.subjects_dir,
              cortex='low_contrast', background='white', size=(800, 600))
brain.add_annotation('aparc_sub')
label_nms = [rr.name for rr in fslabels]

dfs = list()
for aix, age in enumerate(ages):
    with xr.open_dataset(op.join(defaults.datadir,
                                 'genz_%s_degree.nc' % age)) as ds:
        df_ = ds.to_dataframe().reset_index()
        df_['age'] = np.ones(len(df_)) * age
        dfs.append(df_)

Df = pd.concat(dfs)
cols = pd.DataFrame(np.array([re.split('_|-|', ll) for ll in Df.roi]),
                    columns=['label', 'label_ix', 'hem'])
Df = Df.join(cols).reset_index().drop(labels=['index'], axis=1)
ma_ = ['parsopercularis', 'parsorbitalis']
g = sns.FacetGrid(Df[Df.label.isin(ma_)],
                  col='band', row='hem', hue='age', sharey=False,
                  aspect=1.5)
kde = (g.map(sns.distplot, 'degree', rug=True, hist=False).add_legend())

Df_ = Df.set_index(['band', 'roi'], inplace=False).sort_index()
for ii, (kk, vv) in enumerate(bands.items()):
    g = sns.catplot(x='age', y='degree', hue='label', kind='point', height=12,
                    aspect=1, palette='colorblind', col='hem',
                    legend_out=True,
                    data=Df_.loc[(kk)])
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('%d-%dHz' % (vv[0], vv[1]))
    g.savefig(op.join(defaults.figs_dir, 'genz_%s_degrees-curve.png' % kk),
              dpi=140, format='png')
