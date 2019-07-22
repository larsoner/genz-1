#!/usr/bin/env python

"""Global study parameters."""

__author__ = 'Kambiz Tavabi'
__copyright__ = 'Copyright 2018, Seattle, Washington'
__license__ = 'MIT'
__version__ = '0.1.0'
__maintainer__ = 'Kambiz Tavabi'
__email__ = 'ktavabi@uw.edu'
__status__ = 'Development'

from os import path as op
from pathlib import Path

static = op.join(Path(__file__).parents[1], 'static')
datadir = op.join(Path(__file__).parents[1], 'data')
megdir = '/media/ktavabi/ALAYA/data/ilabs/nbwr'
bem_work = op.join(Path(__file__).parents[1], 'processed')
subjects_dir = '/brainstudio/data/genz/freesurf_subjs'
fs_affix = '_ses-1_fsmempr_ti1100_rms_1_freesurf_hires'
bids_affix = 'ses-001_task-lexicaldecision_run-01'
exclude = []

