import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_results
from common import save_figure, figure_absent
from params import stats_dir

from figures import *

OVERWRITE = True


first = True

school_names = [f.stem for f in stats_dir.glob('*.pkl')]

# now put them all together
print('Boxplot for all schools')
from itertools import chain
results = []
for s in school_names:
    mine = load_results(s)
    if mine is None: continue

    mine = [r for r in mine if r is not None]
    for r in mine:
        r['school'] = s

    results.extend(mine)

df_all = pd.DataFrame.from_records(results)

# we need to go through and find the number of freshmen known by upperclassmen in each school
from load_data import load_school
true_updated = {}
for s in school_names:
    A, df = load_school(s)
    freshmen = df[df.year == 2009].index
    upperclassmen = df[(df.year >= 2006) & (df.year <= 2008)].index

    has_upperclassman_friend = A[freshmen, :][:, upperclassmen].sum(axis=1) > 0
    true_updated[s] = int(has_upperclassman_friend.sum())

# replace these in df_all
df_all['true_updated'] = df_all['school'].map(true_updated)

df_all['true'] = df_all['true_updated']

if figure_absent('all', 'boxplot_all') or OVERWRITE:
    df = df_all[df_all['type'] == 'everyone']
    boxplot(df)
    save_figure('all', 'boxplot_all')

    density_plot(df)
    save_figure('all', 'density_all')

if figure_absent('all', 'boxplot_upperclassmen') or OVERWRITE:
    df = df_all[df_all['type'] == 'upperclassmen']
    boxplot(df)
    save_figure('all', 'boxplot_upperclassmen')

    density_plot(df)
    save_figure('all', 'density_upperclassmen')

print(f'Boxplot for {10} schools')
from random import shuffle
for s in school_names[:10]:
    if figure_absent(s, 'boxplot_all') or OVERWRITE:
        rows = load_results(s)
        df_all = pd.DataFrame.from_records(rows)
        df = df_all[df_all['type'] == 'everyone']
        boxplot(df)
        save_figure(s, 'boxplot_all')

    if figure_absent(s, 'boxplot_upperclassmen') or OVERWRITE:
        rows = load_results(s)
        df_all = pd.DataFrame.from_records(rows)
        df = df_all[df_all['type'] == 'upperclassmen']
        boxplot(df)
        save_figure(s, 'boxplot_upperclassmen')

