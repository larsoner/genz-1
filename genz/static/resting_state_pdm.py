# -*- coding: utf-8 -*-

"""Basic expyfun paradigm file for GenZ resting state.
Present white fixation cross on black background for 5 minutes"""

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
#
# License: MIT


import numpy as np

from genz.static.expyfun import ExperimentController
from genz.static.expyfun import Line

# set configuration
dur = 300.0
xy1 = np.array([[0, 0], [-1, 1]])  # vertical line
xy2 = np.array([[-1, 1], [0, 0]])  # horizontal line

instructions = ('For the next 5 minutes you will see a black screen with '
                'a white cross at the center. '
                'Your job is to fix your eyes on the cross for the entire '
                'time. Make sure to sit still, moving as little as possible. '
                'Try to relax and use your imagination to daydream! '
                'Remember to breathe, and blink normally throughout. ')
goodbye = 'You are done! Thank you for your cooperation.'

with ExperimentController('testExp', participant='foo', session='001',
                          output_dir=None, version='fa29bb7') as ec:
    # show instructions
    ec.screen_prompt(instructions, live_keys=[1, 2], max_wait=np.inf)
    line1 = Line(ec, xy1, units='deg', line_color='w', line_width=2.0)
    line2 = Line(ec, xy2, units='deg', line_color='w', line_width=2.0)
    # do the drawing, then flip
    for obj in [line1, line2]:
        obj.draw()
    ec.flip()
    ec.wait_secs(dur)
    ec.screen_prompt(goodbye, live_keys=[9], max_wait=10.0)

