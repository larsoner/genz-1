# -*- coding: utf-8 -*-
"""Write tsv files containing subject id & gender list."""

import pandas as pd
import numpy as np
import seaborn as sns
import pprint

sns.set(style='ticks', color_codes=True)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.set_option('precision', 2)

# Subjects with MEG data ACQ
subjects = [
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

# picks for MEG datasets
ssdf = pd.DataFrame({'id': subjects})
ssdf.sort_values(by='id')

# merge with RA spreadsheets
dfs = []
for ag in [9, 11, 13, 15, 17]:
    tsv = '/home/ktavabi/Github/genz/static/' \
          'GenZ_subject_information - %da group.tsv' % ag
    tmp = pd.read_csv(tsv, sep='\t',
                      usecols=['Subject Number', 'Sex', 'Bad Channels',
                               'Age (Years) at enrollement', 'LPA', 'RPA'],
                      keep_default_na=False)
    tmp.insert(len(tmp.columns), "ag", np.repeat(ag, len(tmp)))
    dfs.append(tmp)

df = pd.concat(dfs, ignore_index=True)
df.columns = ['id', 'sex', 'yrs', 'lpa', 'rpa', 'badChs', 'ag']
df.id = [ii[5:] for ii in df.id.values]
df.sort_values(by='id')
dups = df[df.duplicated('id')].index.values.tolist()
df.drop(df.index[dups], inplace=True)
ssdf.merge(df, on='id', validate='1:1').to_csv(
        '/home/ktavabi/Github/genz/static/picks.tsv', sep='\t')
ssdf = ssdf.merge(df, how='outer', on='id', validate='1:1')

# bit of wrangling
ssdf['id'].replace('', np.nan, inplace=True)
ssdf.dropna(subset=['id'], inplace=True)
assert (np.logical_or(ssdf['sex'].values == 'male',
                      ssdf['sex'].values == 'female')).all()
ssdf.lpa = pd.to_numeric(ssdf.lpa)
ssdf.rpa = pd.to_numeric(ssdf.rpa)
lpas = ssdf[~(ssdf['lpa'].values < 0)].id
ssdf.loc[ssdf.id.isin(lpas), 'lpa'] = - ssdf.loc[ssdf.id.isin(lpas), 'lpa']
ssdf['ag'] = ssdf['ag'].astype('category')
ssdf.yrs = pd.to_numeric(ssdf.yrs)

# spit out descriptives & plot data for subject age & periauricular fids
desc = ssdf.groupby('ag').describe()
pprint.pprint(desc)
g = sns.pairplot(ssdf, vars=['yrs', 'lpa', 'rpa'], hue="sex",
                 palette="husl", diag_kind="kde", height=2.5)

h = sns.catplot(x="ag", hue="sex", data=ssdf, kind="count",
                height=4, aspect=1.7)
(h.set_axis_labels("Age Group", "Count")
 .set(ylim=(7, 19))
 .despine(left=True))

# picks for MRI datasets in list compiled by Neva Corrigan
xls = pd.read_excel('GenzMRISubjectNumbers.xlsx', header=None)
xls['id'] = xls[0]
xls['sex'] = xls.id.map(lambda x: '%s' % x.split('z')[1]).astype(np.int64)
ages = xls.sex.values
xls.sex = xls.sex.map(lambda x: x % 2 == 0).map({True: 'female', False: 'male'})
n = 100
tmp = np.zeros_like(ages)
for ii in np.arange(9, 19, 2):
    tmp[np.where(np.logical_and(ages < n + 100, ages > n))] = ii
    n += 100
xls['tmp'] = tmp
xls.id = xls[['id', 'tmp']].astype(str).apply(lambda x: '_'.join(x), axis=1)
xls.drop([0, 'tmp'], inplace=True, axis=1)
xls.id = xls.id.map(lambda x: '%sa' % x)
xls.to_csv('/home/ktavabi/Github/genz/static/GenzMRISubjectNumbers.tsv',
           sep='\t')
