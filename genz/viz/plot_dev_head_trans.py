# -*- coding: utf-8 -*-

"""Use median absolute deviation to identify outliers in head position obs"""

# Authors: Kambiz Tavabi <ktavabi@uw.edu>

import os.path as op
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.linalg as LA
import mne
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from statsmodels.robust import mad
from genz.picks import names

study_dir = '/media/ktavabi/INDAR/data/genz/rsMEG'
subjects = np.asarray(names)[np.setdiff1d(np.arange(len(names)),
                                          np.array([3]))]
poss = np.zeros((len(subjects), 3))
lp_cutoffs = np.zeros((len(subjects)))
for ii, subj in enumerate(subjects):
    raw_file = glob.glob(op.join(study_dir, subj,
                                 'raw_fif', '*_raw.fif'))[0]
    raw = mne.io.read_raw_fif(raw_file, allow_maxshield='yes')
    poss[ii] = raw.info['dev_head_t']['trans'][:3, 3]
    lp_cutoffs[ii] = raw.info['highpass']
np.savez_compressed(op.join(study_dir, 'initial_head_poss.npz'), poss=poss,
                    subjects=subjects)
poss = np.load(op.join(study_dir, 'initial_head_poss.npz'))['poss']
poss_norm = LA.norm(poss, axis=1)

mad_poss_norm = mad(poss_norm)
'''
    Median Absolute deviation
    R-blboggers - Absolute Deviation Around the Median
    https://www.r-bloggers.com/absolute-deviation-around-the-median/

    Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
    Handle Outliers", The ASQC Basic References in Quality Control:
    Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
'''
# Outliers defined as >/< +/- 3 * MAD
mask = ~np.logical_or(poss_norm > np.median(poss_norm) + 2.5 * mad_poss_norm,
                      poss_norm < np.median(poss_norm) - 2.5 * mad_poss_norm)
# Figure window dressing
sns.set(style="white", palette="colorblind", color_codes=True)
colors = sns.color_palette()
rgb = [tuple([v for v in values]) for values in colors]
box_kwargs = {'showmeans': False, 'meanline': True}
# noinspection PyTypeChecker
flierprops = dict(marker='d', markersize=5, color=rgb[0])

# Plot head pos data
f, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True)
f.suptitle('Head positions relative to device origin', size=14)
sns.distplot(poss_norm * 1000, color=rgb[0], ax=axes[0])
axes[0].set(xlabel='Distance (mm)', ylabel='Frequency')
sns.boxplot(poss_norm * 1000, ax=axes[1], color=rgb[0],
            flierprops=flierprops, **box_kwargs)
h = sns.swarmplot(poss_norm[mask] * 1000, ax=axes[1], color=rgb[3])
h.set(xlabel='Distance (mm)')

# Using Line2D to create the markers for the legend.
# This is the creation of the proxy artists.
line = Line2D(range(1), range(1), color='k')
circle = Line2D(range(1), range(1), color='white', marker='o', markersize=5,
                markerfacecolor=rgb[3])
diamond = Line2D(range(1), range(1), color='white', marker='d', markersize=5,
                 markerfacecolor=rgb[0])
extra = Rectangle((0, 0), .5, 1, fc="w", fill=False, edgecolor='none',
                  linewidth=0)
# Calling the handles and labels to create the legend.
f.legend([extra, line, circle, diamond],
         ['N = %d' % len(subjects),
          'median = %.0f' % np.median(poss_norm * 1000),
          'positions', 'outliers'],
         loc=0, bbox_to_anchor=(1, .9), numpoints=1,
         frameon=False)
sns.despine(left=True)
plt.show()
print('Subjects with initial head position larger than 3 times the '
      'absolute deviation from sample median:')
for subj in np.asarray(subjects)[~mask]:
    print(' %s' % subj)
print('Subjects with highpass above DC:\n')
print(subjects[lp_cutoffs > 0])
