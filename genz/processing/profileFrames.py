# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import xarray as xr
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


@pf.register_dataframe_method
def str_remove(df, column_name: str, pattern: str = ''):
    """Remove string patten from a column

    Wrapper around df.str.replace()

    Parameters
    -----------
    df: pd.Dataframe
        Input dataframe to be modified
    column_name: str
        Name of the column to be operated on
    pattern: str, default to ''
        String pattern to be removed

    Returns
    --------
    df: pd.Dataframe

    """
    df[column_name] = df[column_name].str.replace(pattern, '')
    return df


# %%
@pf.register_dataframe_method
def str_slice(df, column_name: str, start: int = 0, stop: int = -1):
    """Slice a column of strings by indexes

    Parameters
    -----------
    df: pd.Dataframe
        Input dataframe to be modified
    column_name: str
        Name of the column to be operated on
    start: int, optional, default to 0
        Staring index for string slicing
    stop: int, optional, default to -1
        Ending index for string slicing

    Returns
    --------
    df: pd.Dataframe

    """
    df[column_name] = df[column_name].str[start:stop]
    return df


# %%
@pf.register_dataframe_method
def str_word(
    df,
    column_name: str,
    start: int = None,
    stop: int = None,
    pat: str = " ",
    *args,
    **kwargs
):
    """
    Wrapper around `df.str.split` with additional `start` and `end` arguments
    to select a slice of the list of words.

    :param df: A pandas DataFrame.
    :param column_name: A `str` indicating which column the split action is to be made.
    :param start: optional An `int` for the start index of the slice
    :param stop: optinal  An `int` for the end index of the slice
    :param pat: String or regular expression to split on. If not specified, split on whitespace.

    """
    df[column_name] = df[column_name].str.split(pat).str[start:stop]
    return df



# %%
idcol = ['id']
features = ['gender',
            'peermindset',
            'persmindset',
            'needforapproval',
            'needforbelonging',
            'rejection',
            'coping_mad',
            'coping_sad',
            'coping_worried',
            'rsqanxiety',
            'rsqanger',
            'cdimean',
            'moodgood',
            'moodhappy',
            'moodrelaxed',
            'stateanxiety',
            'traitanxiety']

# %%
bsdf = (
    pd.read_excel(op.join(defaults.static, 'Mlab-ASOSt1.xlsx'))
    .clean_names()
    .str_remove('id', pattern='GenZ_')
    .remove_empty()
    .select_columns(idcol + features)
    .encode_categorical(['gender'])
    .reset_index(drop=True)
)

bsdf.head(3)


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
    for cn in ['age_months_at_enrollement', 'age_years_at_enrollement']:
        tmp[cn] = pd.to_numeric(tmp[cn], errors='coerce')
    tmp.insert(len(tmp.columns), 'age', np.repeat(age, len(tmp)))
    dfs.append(tmp)
df = pd.concat(dfs, ignore_index=True)
df = (df.clean_names()
    .rename_columns({"subject_number": "id"})
    .str_remove('id', pattern='GenZ_')
)
df.head(3)

# %%
# picks from resting MEG datasets
temp = pd.DataFrame({'id': defaults.picks})
temp = temp.join(df.set_index('id'), on='id', how='inner')

data = (
    temp.join(bsdf.set_index('id'), on='id', how='inner')
    .clean_names()
    .encode_categorical(['age', 'gender']))
data.head(3)


# %%
tmp = list()
for aix, age in enumerate(defaults.ages):
    with xr.open_dataset(op.join(defaults.payload,
                                 'genz_%s_degree.nc' % age)) as ds:
        df_ = ds.to_dataframe().reset_index()
        df_['age'] = np.ones(len(df_)) * age
        tmp.append(df_)
df = pd.concat(tmp)
df = (
    df.clean_names()
    .str_remove('id', pattern='genz')
    .reset_index(drop=True)
)
df['hemisphere'] = df['roi'].str[-2:]

df.head(3)


# %%
df_final = (
    pd.merge(
        data, df, how='right', on='id'
    )
    .drop_duplicates(keep='first')
    .str_slice('roi', stop=-3)
    .reset_index(drop=True)
    .sort_naturally('id')
)

df_final.head(3)
df_final.to_csv(op.join(defaults.payload, 'nxDegree-ASOSt1.csv'))

# %%
# kwargs = {'interactions': {'targets': features}}
report = df_final.profile_report(title='Data profile',
                                 minimal=False, explorative=True,
                                 sensitive=False, dark_mode=False,
                                 orange_mode=True,
                                 config_file=op.join(
                                     defaults.processing, 'profiler-config.yaml'),
                                 )

#report.to_file(op.join(defaults.payload, 'nxDegree-ASOSt1_profile.html'))
