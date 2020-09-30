# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import os.path as op
import pprint

import janitor
import numpy as np
import pandas as pd
import pandas_flavor as pf
import seaborn as sns
from pandas_profiling import ProfileReport

from genz import defaults


# %%
sns.set(style='ticks', color_codes=True)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.set_option('precision', 2)


# %%
cols = ['peermindset',
        'persmindset',
        'needforapproval',
        'needforbelonging',
        'Rejection',
        'Coping_mad',
        'Coping_sad',
        'Coping_worried',
        'RSQanxiety',
        'RSQanger',
        'CDImean',
        'Moodgood',
        'Moodhappy',
        'Moodrelaxed',
        'StateAnxiety',
        'TraitAnxiety']
dtypes = {k: np.int for k in cols}
cols.extend(['gender', 'ID'])


# %%
bsdf = (
    pd.read_excel(op.join(defaults.static, 'Mlab-ASOSt1.xlsx'),
                  usecols=cols, dtype=dtypes)
    .clean_names()
    .remove_empty()
    .encode_categorical(['gender'])
    .drop(columns=[])
    .reset_index(drop=True)
)
bsdf['id'] = [ii[5:] for ii in bsdf['id'].values]

bsdf.head()


# %%
# merge with RA spreadsheets
dfs = []
for age in defaults.ages:
    fi = op.join(defaults.static,
                 'GenZ_subject_information - %da group.tsv' % age)
    tmp = (
        pd.read_csv(fi, sep='\t',
                    usecols=['Subject Number', 'Sex',
                             'Age (Years) at enrollement',
                             'Age (Months) at enrollement'
                             ],
                    keep_default_na=False)
        .clean_names())
    tmp.insert(len(tmp.columns), 'age', np.repeat(age, len(tmp)))
    dfs.append(tmp)
df = pd.concat(dfs, ignore_index=True)
for cn in ['age_months_at_enrollement', 'age_years_at_enrollement']:
    df[cn] = pd.to_numeric(df[cn], errors='coerce')
df['id'] = [ii[5:] for ii in df.subject_number]
df.profile_report().to_widgets()


# %%
# picks from resting MEG datasets
temp = pd.DataFrame({'id': defaults.picks})
temp = temp.join(df.set_index('id'), on='id', how='inner')

data = (
    temp.join(bsdf.set_index('id'), on='id', how='inner')
    .clean_names()
    .encode_categorical(['age', 'gender']))

report = bsdf.profile_report(title='Social measures profile',
                             minimal=False, explorative=True,
                             sensitive=False, dark_mode=False,
                             orange_mode=True,
                             config_file=op.join(
                                 defaults.processing, 'profiler-config.yaml'),
                             )
report.to_file(op.join(defaults.payload, 'ASOSt1_profile.html'))
data.head()

# %%
