# %% [markdown]
# # Network Size Estimation with Degree Bias Correction - Results Viewer
# 
# This notebook provides an interactive way to explore the results of our network size estimation study. It includes pre-computed results for all schools in the Facebook100 dataset, with various estimation methods compared.
# 
# ## Setup
# 
# First, let's clone the repository and install dependencies:

# %%
!git clone https://github.com/scholar-anon/joint-scale-up.git
!cd joint-scale-up && pip install -r requirements.txt

# %% [markdown]
# ## Import Required Libraries

# %%
import sys
sys.path.append('joint-scale-up')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import os

# %% [markdown]
# ## Load Pre-computed Results

# %%
def load_school(fn):
    with open(fn, 'rb') as f:
        rows = pickle.load(f)
        for r in rows:
            r['school'] = os.path.basename(fn).split('_stats.pkl')[0]
        return rows

stats_dir = Path('joint-scale-up/stats')
results = {}

for fn in stats_dir.glob('*.pkl'):
    school_name = fn.stem
    results[school_name] = load_school(fn)

print(f"Loaded results for {len(results)} schools")

# %% [markdown]
# ## Interactive Results Viewer

# %%
def plot_school_results(school_name):
    if school_name in results:
        school_data = results[school_name]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Boxplot of estimates
        estimates = pd.DataFrame(school_data)
        sns.boxplot(data=estimates, y='method', x='estimate', ax=ax1)
        ax1.set_title(f'Estimates for {school_name}')
        ax1.set_xlabel('Estimated Size')
        
        # Plot 2: Error distribution
        sns.boxplot(data=estimates, y='method', x='relative_error', ax=ax2)
        ax2.set_title(f'Relative Error for {school_name}')
        ax2.set_xlabel('Relative Error')
        
        plt.tight_layout()
        plt.show()
    else:
        print(f"No results found for {school_name}")

from ipywidgets import interact, widgets

school_names = sorted(list(results.keys()))
interact(plot_school_results, school_name=school_names)

# %% [markdown]
# ## Comparison of All Methods

# %%
def plot_all_schools_comparison():
    # Combine all results into a single DataFrame
    all_data = []
    for school_name, school_data in results.items():
        for row in school_data:
            row['school'] = school_name
            all_data.append(row)
    
    df = pd.DataFrame(all_data)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Boxplot of estimates by method
    sns.boxplot(data=df, y='method', x='estimate', ax=ax1)
    ax1.set_title('Estimates Across All Schools')
    ax1.set_xlabel('Estimated Size')
    
    # Plot 2: Boxplot of relative errors by method
    sns.boxplot(data=df, y='method', x='relative_error', ax=ax2)
    ax2.set_title('Relative Error Across All Schools')
    ax2.set_xlabel('Relative Error')
    
    plt.tight_layout()
    plt.show()

plot_all_schools_comparison()

# %% [markdown]
# ## Summary Statistics

# %%
def compute_summary_statistics():
    # Combine all results into a single DataFrame
    all_data = []
    for school_name, school_data in results.items():
        for row in school_data:
            row['school'] = school_name
            all_data.append(row)
    
    df = pd.DataFrame(all_data)
    
    # Compute summary statistics by method
    summary = df.groupby('method').agg({
        'estimate': ['mean', 'std', 'min', 'max'],
        'relative_error': ['mean', 'std', 'min', 'max']
    }).round(3)
    
    return summary

summary_stats = compute_summary_statistics()
display(summary_stats) 