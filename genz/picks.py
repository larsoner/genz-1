# -*- coding: utf-8 -*-
"""Script generates tsv files with lists of subjects id and
gender for MR and MEG modalities.
"""

import pandas as pd
import numpy as np

 # Too few EOG events
# picks for MEG datasets
ssdf = pd.DataFrame({'id': subjects})
ssdf.sort_values(by='id')



df = pd.concat(dfs, ignore_index=True)
ssdf.merge(df, on='id', validate='1:1').to_csv(
    '/home/ktavabi/Github/genz/static/picks.csv')
/home/ktavabi/Github/genz/genz/static/GenZ_subject_information - 9a group.tsv