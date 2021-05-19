#!/usr/bin/env python
# Authors: Kambiz Tavabi <ktavabi@gmail.com>
#
# License: MIT

"""Global study parameters."""
# TODO convert to YAML

from os import path as op
from pathlib import Path

import numpy as np

static = op.join(Path(__file__).parents[0], "static")
processing = op.join(Path(__file__).parents[0], "processing")
payload = op.join(Path(__file__).parents[0], "payload")
archive = op.join(Path(__file__).parents[0], "archive")

megdata = "/media/ktavabi/INDAR/data/ilabs/genz_resting/"
subjects_dir ="/media/ktavabi/INDAR/data/ilabs/anatomy" 
exclude = [  # these are unusable
    "113_9a",  # unusable
    "115_9a",  # missing resting recording
    "318_13a",  # missing resting recording
    "319_13a",  # unusable
]
n_jobs = 'cuda'
# frequencies of interest
bands = {
    "DC": (0, 1.9),
    "delta": (2, 4),
    "theta": (5, 7),
    "alpha": (8, 12),
    "beta": (13, 29),
    "gamma": (30, 49),
    "gamma2": (50, 80)
}
# ages
ages = np.arange(9, 19, 2)

# Down sample data to
new_sfreq = 250.0

#
cmap_limits = [75, 85, 95]

# Subjects with MEG data ACQ
include = [
    "105_9a",
    "106_9a",
    "108_9a",
    "110_9a",
    "111_9a",
    "112_9a",
    "113_9a",
    "114_9a",
    "115_9a",
    "116_9a",
    "118_9a",
    "119_9a",
    "120_9a",
    "121_9a",
    "122_9a",
    "123_9a",
    "124_9a",
    "125_9a",
    "126_9a",
    "128_9a",
    "130_9a",
    "131_9a",
    "133_9a",
    "205_11a",
    "206_11a",
    "207_11a",
    "208_11a",
    "209_11a",
    "210_11a",
    "212_11a",
    "216_11a",
    "218_11a",
    "219_11a",
    "223_11a",
    "224_11a",
    "225_11a",
    "226_11a",
    "227_11a",
    "228_11a",
    "229_11a",
    "230_11a",
    "231_11a",
    "232_11a",
    "233_11a",
    "235_11a",
    "311_13a",
    "313_13a",
    "314_13a",
    "318_13a",
    "319_13a",
    "320_13a",
    "322_13a",
    "323_13a",
    "324_13a",
    "325_13a",
    "326_13a",
    "327_13a",
    "328_13a",
    "329_13a",
    "330_13a",
    "331_13a",
    "332_13a",
    "333_13a",
    "334_13a",
    "335_13a",
    "337_13a",
    "401_15a",
    "403_15a",
    "406_15a",
    "409_15a",
    "411_15a",
    "412_15a",
    "413_15a",
    "414_15a",
    "415_15a",
    "417_15a",
    "418_15a",
    "419_15a",
    "420_15a",
    "421_15a",
    "422_15a",
    "423_15a",
    "424_15a",
    "425_15a",
    "426_15a",
    "427_15a",
    "428_15a",
    "429_15a",
    "430_15a",
    "431_15a",
    "432_15a",
    "507_17a",
    "512_17a",
    "513_17a",
    "514_17a",
    "515_17a",
    "516_17a",
    "517_17a",
    "518_17a",
    "519_17a",
    "520_17a",
    "521_17a",
    "522_17a",
    "523_17a",
    "527_17a",
    "528_17a",
    "529_17a",
    "530_17a",
    "531_17a",
    "532_17a",
]  # List compiled by Erica Peterson
