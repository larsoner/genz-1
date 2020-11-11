#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Global study parameters."""

from os import path as op
from pathlib import Path

import numpy as np

static = op.join(Path(__file__).parents[0], 'static')
processing = op.join(Path(__file__).parents[0], 'processing')
payload = op.join(Path(__file__).parents[0], 'payload')
archive = op.join(Path(__file__).parents[0], 'archive')
megdata = '/media/ktavabi/ALAYA/data/ilabs/genz_resting'
subjects_dir = '/media/ktavabi/ALAYA/data/freesurfer'
exclude = ['104_9a',  # Too few EOG events
           '108_9a',  # Fix
           '113_9a',  # Too few ECG events
           '115_9a',  # no cHPI
           '121_9a',  # Source space missing points
           '209_11a',  # Too few EOG events
           '231_11a',  # twa_hp calc fail with assertion error
           '427_15a',  # no epochs
           '432_15a',  # Too few ECG events
           '510_17a',  # Too few ECG events
           '527_17a']  # Too few EOG events

# frequencies of interest
bands = {'beta': (13, 29), 'gamma': (30, 80)
    }

# ages
ages = np.arange(9, 19, 2)

# Down sample data to
new_sfreq = 300.

# Subjects with MEG data ACQ
include = [
    '104_9a',
    '105_9a',
    '106_9a',
    '108_9a',
    '109_9a',
    '110_9a',
    '111_9a',
    '112_9a',
    '113_9a',
    '114_9a',
    '115_9a',
    '116_9a',
    '117_9a',
    '118_9a',
    '119_9a',
    '120_9a',
    '121_9a',
    '122_9a',
    '123_9a',
    '124_9a',
    '125_9a',
    '126_9a',
    '128_9a',
    '129_9a',
    '130_9a',
    '131_9a',
    '132_9a',
    '133_9a',
    '201_11a',
    '202_11a',
    '203_11a',
    '205_11a',
    '206_11a',
    '207_11a',
    '208_11a',
    '209_11a',
    '210_11a',
    '211_11a',
    '212_11a',
    '213_11a',
    '214_11a',
    '216_11a',
    '218_11a',
    '219_11a',
    '220_11a',
    '221_11a',
    '222_11a',
    '223_11a',
    '224_11a',
    '225_11a',
    '226_11a',
    '227_11a',
    '228_11a',
    '229_11a',
    '230_11a',
    '231_11a',
    '232_11a',
    '233_11a',
    '235_11a',
    '302_13a',
    '305_13a',
    '307_13a',
    '308_13a',
    '309_13a',
    '310_13a',
    '311_13a',
    '312_13a',
    '313_13a',
    '314_13a',
    '316_13a',
    '319_13a',
    '320_13a',
    '321_13a',
    '322_13a',
    '323_13a',
    '324_13a',
    '325_13a',
    '326_13a',
    '327_13a',
    '328_13a',
    '329_13a',
    '330_13a',
    '331_13a',
    '332_13a',
    '333_13a',
    '334_13a',
    '335_13a',
    '337_13a',
    '401_15a',
    '402_15a',
    '403_15a',
    '404_15a',
    '406_15a',
    '408_15a',
    '410_15a',
    '411_15a',
    '412_15a',
    '413_15a',
    '414_15a',
    '415_15a',
    '416_15a',
    '417_15a',
    '418_15a',
    '419_15a',
    '420_15a',
    '421_15a',
    '422_15a',
    '423_15a',
    '424_15a',
    '425_15a',
    '426_15a',
    '427_15a',
    '428_15a',
    '432_15a',
    '501_17a',
    '502_17a',
    '503_17a',
    '504_17a',
    '505_17a',
    '506_17a',
    '507_17a',
    '508_17a',
    '509_17a',
    '510_17a',
    '511_17a',
    '512_17a',
    '513_17a',
    '514_17a',
    '515_17a',
    '516_17a',
    '517_17a',
    '518_17a',
    '519_17a',
    '520_17a',
    '521_17a',
    '522_17a',
    '523_17a',
    '524_17a',
    '525_17a',
    '526_17a',
    '527_17a',
    '528_17a',
    '529_17a',
    '530_17a',
    '531_17a',
    '532_17a'
]  # List compiled by Erica Peterson

exclude = ['104_9a',  # Too few EOG events
           '108_9a',  # Fix
           '113_9a',  # Too few ECG events
           '115_9a',  # no cHPI
           '209_11a',  # Too few EOG events
           '231_11a',  # twa_hp calc fail with assertion error
           '432_15a',  # Too few ECG events
           '510_17a', ] 