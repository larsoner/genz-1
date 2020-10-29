# %%
import os.path as op

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pandas_flavor as pf
import janitor

import seaborn as sns
import xarray as xr
from genz import defaults
from numpy import mean, std
from pandas_profiling import ProfileReport
from sklearn import datasets, svm
from sklearn.compose import make_column_transformer
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import (HistGradientBoostingRegressor,
                              RandomForestRegressor, StackingRegressor)
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LassoCV, RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils import shuffle

# %%
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.set_option('precision', 2)
seed = np.random.seed(42)


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


@pf.register_dataframe_method
def explode(df: pd.DataFrame, column_name: str, sep: str):
    """
    For rows with a list of values, this function will create new rows for each value in the list

    :param df: A pandas DataFrame.
    :param column_name: A `str` indicating which column the string removal action is to be made.
    :param sep: The delimiter. Example delimiters include `|`, `, `, `,` etc.
    """

    df["id"] = df.index
    wdf = (
        pd.DataFrame(df[column_name].str.split(sep).fillna("").tolist())
        .stack()
        .reset_index()
    )
    # exploded_column = column_name
    wdf.columns = ["id", "depth", column_name]  # plural form to singular form
    # wdf[column_name] = wdf[column_name].apply(lambda x: x.strip())  # trim
    wdf.drop("depth", axis=1, inplace=True)

    return pd.merge(df, wdf, on="id", suffixes=("_drop", "")).drop(
        columns=["id", column_name + "_drop"]
    )


# %%
asos = (
    pd.read_excel(op.join(defaults.static, 'Mlab-ASOSt1.xlsx'))
    .clean_names()
    .str_remove('id', pattern='GenZ_')
    .remove_empty()
    .select_columns(['id','gender',
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
            'traitanxiety'])
    .encode_categorical(['gender'])
    .set_index('id')
)

asos.head(3)

# %%
mapping = {-9999.0: np.nan, 0.0: np.nan}
degrees = (
    pd.read_csv(op.join(defaults.payload, 'degree_x_frequency-roi.csv'))
    .str_remove('id', pattern='GenZ_')
    .clean_names()
    .remove_columns(['unnamed_0'])
    .deconcatenate_column(
         column_name='roi', new_column_names=['label', 'hemi'],
         sep='-', preserve_position=True
     )
    .find_replace(
        deg=mapping)
    .dropna()
    .impute(column_name='deg', statistic_column_name='mean')
    .rename_columns({'freq': 'nx', 'label': 'node',})
    .label_encode(column_names=['nx', 'node', 'hemi'])
    .encode_categorical(columns=['nx', 'node', 'hemi'])
    .sort_naturally("id")
    .set_index('id')
    )

degrees.head()


# %%
# social
behavior = (
    pd.read_excel(op.join(defaults.static, 'behavior.xlsx'), sheet_name='behavior',
    header=None)
    .row_to_names(0)
    .clean_names()
    .str_remove('id', pattern='GenZ_')
    .dropna()
)
behavior.head()

# %%
social = (
    pd.read_excel(op.join(defaults.static, 'behavior.xlsx'), sheet_name='social',
    header=None)
    .row_to_names(0)
    .clean_names()
    .str_remove('id', pattern='GenZ_')
    .dropna()
)

behavior_soc = behavior.set_index('id').merge(social.set_index('id'), right_index=True, left_index=True)
behavior_soc.head(10)
# %%
# merge with RA spreadsheets & MEG frames
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
df_ = (
    pd.concat(dfs, ignore_index=True)
    .clean_names()
    .rename_columns({"subject_number": "id", 'age_months_at_enrollement': 'mos', 'age_years_at_enrollement': 'yrs'})
    .str_remove('id', pattern='GenZ_')
    .set_index('id')
)

df = df_.merge(degrees,right_index=True, left_index=True)
meg_asos = df.merge(asos, right_index=True, left_index=True)
data = meg_asos.merge(behavior_soc, right_index=True, left_index=True)
data.sample(15)
data.to_csv(op.join(defaults.payload, 'data.csv'))

# %%
report = data.profile_report(title='Data profile',
                             minimal=False, explorative=True,
                             sensitive=False, dark_mode=False,
                             orange_mode=True,
                             config_file=op.join(
                                 defaults.processing, 'profiler-config.yaml'),
                             )
report.to_file(op.join(defaults.payload, 'data_profile.html'))
# %%
