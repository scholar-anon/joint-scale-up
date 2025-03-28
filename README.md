# Network Size Estimation with Degree Bias Correction

This project implements and compares various methods for estimating network sizes while accounting for degree bias in social networks. The implementation is tested on the Facebook100 dataset, focusing on estimating the number of connections between different student cohorts.

## Quick Start with Google Colab

The easiest way to explore the results is through our Google Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scholar-anon/joint-scale-up/blob/main/code/colab_viewer.ipynb)

This notebook provides:
- Interactive visualization of results for each school
- Comparison plots of different estimation methods
- Summary statistics
- No setup required - just click the link above

## Overview

The project implements several network size estimation methods:
- Joint-Referrals (Authors, 2025)
- Scale-Up (Killworth et al., 1998)
- Plug-in Scale-Up
- Lower Bound (Lanumteang et al., 2011)
- Chao (1987)
- Dombrowski et al. (2012)
- Naive estimation

## Project Structure

- `scaleup.py`: Core implementation of the ScaleUp estimator and various estimation methods
- `stats.py`: Functions for computing network statistics and processing school data
- `figures.py`: Visualization functions for comparing estimation methods
- `01 run all statistics.py`: Main script for processing all schools in parallel
- `02 boxplot comparison.py`: Script for generating comparison plots
- `load_data.py`: Data loading utilities for the Facebook100 dataset
- `params.py`: Configuration parameters

## Key Features

- Parallel processing of multiple schools with memory management
- Caching of results to avoid recomputation
- Multiple estimation methods for comparison
- Visualization tools for analyzing results
- Support for both directed and undirected networks

## Usage

The project includes pre-computed results for all schools in the `stats` folder. You can directly analyze these results without running the full computation:

```bash
python code/02\ boxplot\ comparison.py
```

Note: Running the full statistics computation (`01 run all statistics.py`) is optional and requires significant computational resources:
- At least 32GB of RAM
- Several hours of computation time
- Results are already available in the `stats` folder

## Data

The project uses the [Facebook100 dataset](http://www.archive.org/details/oxford-2005-facebook-matrix), focusing on:
- Freshman class (graduation year 2009)
- Upperclassmen (graduation years 2006-2008)
- Analysis of connections between these cohorts

After downloading and uncompressing the Facebook100 dataset, you'll need to set the `base_dir` parameter in `params.py` to point to the directory containing the dataset files. For example:

```python
base_dir = Path("/path/to/facebook100")
```

The dataset contains:
- Individual `.mat` files for each school's network data
- A copy of the paper "Social Structure of Facebook Networks" by Traud, Mucha, and Porter (2011)
- A README file with additional information about the dataset

## Estimation Methods

The project implements several estimation methods to compare their performance:

1. **Joint-Referrals**: A novel method that accounts for degree bias through joint reporting
2. **Scale-Up**: The classic Killworth et al. (1998) method
3. **Lower Bound**: Based on Lanumteang et al. (2011)
4. **Chao**: The Chao (1987) estimator
5. **Dombrowski**: Implementation of Dombrowski et al. (2012)
6. **Plug-in Scale-Up**: A modified version of the Scale-Up method
7. **Naive**: Simple proportion-based estimation

## Visualization

The project generates several types of visualizations:
- Boxplots comparing different estimation methods
- Density plots showing the distribution of estimates
- Separate plots for "everyone" and "upperclassmen" analyses
- Individual school plots for detailed analysis

## Memory Management

The implementation includes sophisticated memory management:
- Parallel processing with controlled number of concurrent processes
- Memory usage monitoring and process termination if memory usage is too high
- Garbage collection and memory cleanup between operations
- Caching of results to avoid recomputation

## Dependencies

- numpy
- pandas
- matplotlib
- scipy
- networkx
- tqdm
- psutil

## License

[Add appropriate license information] 