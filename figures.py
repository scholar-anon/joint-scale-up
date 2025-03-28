import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

"""
measure_names = {
    'dombrowski': 'Dombrowski et al. (2012)',
    'lanumteang 2011': 'Lanumteang et al. (2011)',
    'scale-up': 'Scale-Up',
    'chao 1987': 'Chao (1987)',
    'joint-mention': 'Authors (2025)'
}

order = ['joint-mention', 'scale-up', 'dombrowski', 'lanumteang 2011', 'chao 1987']
"""

measure_names = {
    'dombrowski': 'Dombrowski et al. (2012)',
    'lanumteang 2011': 'Lower Bound',
    'MLE': 'Scale-Up',
    'PIMLE': 'Plug-in Scale-Up',
    'chao 1987': 'Chao (1987)',
    'joint-mention': 'Joint-Referrals',
    'naive': 'Naive'
}

#order = ['joint-mention', 'MLE', 'PIMLE', 'naive', 'dombrowski', 'lanumteang 2011', 'chao 1987']
order = ['joint-mention', 'MLE', 'lanumteang 2011']
def boxplot(df, filter_ix=None):
  if filter_ix is None:
    filter_ix = df.index >= 0

  measures = df.measure.unique()
  measures = [m for m in measures if m in order]
  measures = sorted(measures, key=lambda x: -order.index(x))

  # compare these using log, as above
  log_values = {}
  for measure_name in measures:
    ix = filter_ix & (df.measure == measure_name) & (~np.isinf(df.value)) & (~np.isnan(df.value)) & (df.value != 0)
    vals = df[ix].value
    true = df[ix].true
    log_values[measure_name] = np.log2(vals / true)

  # remove measures with no values
  measures = [m for m in measures if len(log_values[m]) > 0]

  ydim = len(measures)*2 if len(measures) <= 3 else len(measures)
  xdim = 12

  plt.figure(figsize=(xdim, ydim))
  
  # Create boxplot
  vals = [log_values[m] for m in measures]
  labels = [measure_names[m] for m in measures]
  plt.boxplot(vals, tick_labels=labels, vert=False, flierprops={'marker': '.', 'markersize': 2, 'alpha': 0.5},
        whis=[2, 98])

  # jitter the points if there are more than...
  if len(vals[0]) > 1000:
    for l in plt.gca().lines:
      if l.get_marker() != '':
        ys = l.get_ydata()
        ys += np.random.uniform(-0.1, 0.1, len(ys))
        l.set_ydata(ys)

  plt.axvline(0, color='black', linestyle='--')

  xlim = plt.xlim()

  ticks = range(-5, 6)
  labels = [f'{int(100 * (2**x))}%' if x!=0 else 'True' for x in ticks]
  plt.xticks(ticks, labels)

  max_xlim = max(-x for x in xlim if x < 0)
  plt.xlim(-max_xlim, max_xlim)

  plt.xlabel('Percent of True Value')
  plt.tight_layout()

  # Set font sizes after plotting
  plt.gca().tick_params(axis='both', which='major', labelsize=16)
  plt.gca().xaxis.label.set_size(16)
  plt.gca().yaxis.label.set_size(16)

  return plt

def density_plot(df, filter_ix=None):
    if filter_ix is None:
        filter_ix = df.index >= 0

    measures = df.measure.unique()
    measures = [m for m in measures if m in order]
    measures = sorted(measures, key=lambda x: -order.index(x))

    # compare these using log, as above
    log_values = {}
    for measure_name in measures:
        ix = filter_ix & (df.measure == measure_name) & (~np.isinf(df.value)) & (~np.isnan(df.value)) & (df.value != 0)
        vals = df[ix].value
        true = df[ix].true
        log_values[measure_name] = np.log2(vals / true)

    # remove measures with no values
    measures = [m for m in measures if len(log_values[m]) > 0]

    ydim = len(measures)
    xdim = 12

    plt.figure(figsize=(xdim, ydim))
    
    # Set font sizes
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12

    # Create density plots
    for i, measure in enumerate(measures):
        values = log_values[measure]
        # Create KDE
        kde = stats.gaussian_kde(values)
        # Create x points for the density plot
        x = np.linspace(min(values), max(values), 100)
        # Calculate density
        density = kde(x)
        # Normalize density to fit in the plot
        density = density / density.max() * 0.8  # Scale to 80% of the height
        # Plot the density
        plt.fill_between(x, i, i + density, color='gray', alpha=0.3)
        plt.plot([0, 0], [i+0.5, i+1.5], 'k--', alpha=0.5)  # Vertical line at 0
        plt.text(0.1, i+0.5, measure_names[measure], va='center', fontsize=12)

    plt.axvline(0, color='black', linestyle='--')

    xlim = plt.xlim()

    ticks = range(-5, 6)
    labels = [f'{int(100 * (2**x))}%' if x!=0 else 'True' for x in ticks]
    plt.xticks(ticks, labels)

    max_xlim = max(-x for x in xlim if x < 0)
    plt.xlim(-max_xlim, max_xlim)

    plt.ylim(0, ydim)

    plt.xlabel('Percent of True Value')
    plt.ylabel('')
    plt.yticks([])  # Remove y-axis ticks since we're using text labels
    plt.tight_layout()

    return plt
