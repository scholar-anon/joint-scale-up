"""
Process Facebook100 school data to compute various network statistics.
This script handles parallel processing of multiple schools with memory management
and result caching.
"""

import gc
import logging
import multiprocessing
import os
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
from tqdm import tqdm
import pandas as pd

from load_data import *
from scaleup import ScaleUp, Sample

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
FRESHMAN_GRAD = 2009
SENIOR_GRAD = 2006
SUPER_SENIOR_GRAD = 2005
UPPERCLASSMEN = [2006, 2007, 2008]
YEARS = [FRESHMAN_GRAD] + UPPERCLASSMEN

def dombrowski(samp: Sample, gi: int):
    """
    Compute Dombrowski's method for estimating network size.
    
    Args:
        samp: Sample object containing network data
        gi: Group index to analyze
        
    Returns:
        Estimated size or None if computation fails
    """
    su = samp.su
    s = set(samp.s)
    g = set(samp.su.groups[0])

    # Union of all egonets
    a = set()
    for si in samp.s:
        a |= su.egonet[0][si]

    N_matches = len(a & g & s)
    N_assays = len(a & g)
    
    return len(s & g) * N_assays / N_matches if N_matches > 0 else None

# Define measurement functions
MEASURES = {
    'joint-mention': lambda x, gi: x.JointReferral(gi, 0),
    'chao 1987': lambda x, gi: x.Chao1987(0),
    'PIMLE': lambda x, gi: x.PIMLE([gi])[0],
    'MLE': lambda x, gi: x.MLE([gi])[0],
    'dombrowski': lambda x, gi: dombrowski(x, 0),
    'lanumteang 2011': lambda x, gi: x.Lanumteang2011(0),
    'naive': lambda x, gi: len(x.su.senders) * len(set(x.s) & x.su.groups[0]) / len(x.s), # what proportion of the group is in the sample?
}

def produce_stats(s: ScaleUp, debug: bool = False, sample_size: int = 250) -> List[Dict]:
    """
    Generate statistics for a given ScaleUp object.
    
    Args:
        s: ScaleUp object to analyze
        debug: Whether to show progress bars
        
    Returns:
        List of dictionaries containing measurement results
    """
    rows = []
    rng = tqdm(range(50)) if debug else range(50)

    for _ in rng:
        for gi in range(1, len(YEARS)):
            for measure_name, measure in MEASURES.items():
                try:
                    samp = Sample(s, sample_size)
                    result = measure(samp, gi)
                    rows.append({
                        'measure': measure_name,
                        'grade': gi,
                        'value': result,
                        'true': len(s.groups[0]),
                        'year': YEARS[gi]
                    })
                except Exception as e:
                    logger.warning(f"Error computing {measure_name} for grade {gi}: {e}")
                    rows.append({
                        'measure': measure_name,
                        'grade': gi,
                        'value': None,
                        'true': len(s.groups[0]),
                        'year': YEARS[gi]
                    })

    return rows

def get_memory_usage() -> float:
    """Get current memory usage percentage."""
    return psutil.virtual_memory().percent

def is_memory_stable(threshold: float = 50, duration: int = 60, checks: int = 3) -> Tuple[bool, float]:
    """
    Check if memory usage stays below threshold for specified duration.
    
    Args:
        threshold: Memory usage threshold percentage
        duration: Time to monitor in seconds
        checks: Number of checks to perform
        
    Returns:
        True if memory usage is stable below threshold
    """
    for _ in range(checks):
        mem = get_memory_usage()
        if mem > threshold:
            return False, mem
        time.sleep(duration / checks)
    return True, mem

def process_school(school_name: str, debug: bool = False, cache_only: bool = False, force_recompute: bool = False, sample_size: int = 250) -> Optional[List[Dict]]:
    """
    Process a single school's data and compute statistics.
    
    Args:
        school_name: Name of the school to process
        debug: Whether to show debug information
        cache_only: Whether to only check cache without processing
        force_recompute: Whether to force recomputation of results
        sample_size: Number of samples to sample from the network
    Returns:
        List of dictionaries containing results or None if cache_only and no cache exists
    """
    # Setup cache
    from params import stats_dir
    cache_dir = stats_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_file = cache_dir / f'{school_name}.csv'
    pickle_file = cache_dir / f'{school_name}.pkl'
    
    # Check cache first
    if not force_recompute:
        if csv_file.exists():
            if debug:
                logger.info(f'Loading cached results for {school_name}')
            try:
                df = pd.read_csv(csv_file, sep='\t')
                return df.to_dict('records')
            except Exception as e:
                logger.error(f"Error loading cache for {school_name}: {e}")
        elif pickle_file.exists():
            if debug:
                logger.info(f'Loading legacy pickle results for {school_name}')
            try:
                with pickle_file.open('rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading pickle cache for {school_name}: {e}")
    
    if cache_only:
        return None
    
    # Wait for system memory to be available
    while psutil.virtual_memory().percent >= 50:
        gc.collect()
        time.sleep(random.uniform(30, 90))

    try:
        # load the data
        A, df = load_school(school_name)

        # Prepare groups
        groups = [
            df[df.year == year].index
            for year in YEARS
        ]

        if debug:
            logger.info(f'Producing ScaleUp object for {school_name}')

        # Process everyone
        A = A.toarray() # convert to (dense) numpy array
        s = ScaleUp(A, groups, [0]) # setting the "known" group does nothing in this case

        if debug:
            logger.info(f'Producing everyone stats for {school_name}')

        everyone = produce_stats(s, debug=debug, sample_size=sample_size)
        for x in everyone:
            x['type'] = 'everyone'

        # Free memory from first ScaleUp object
        del s
        gc.collect()

        # Process upperclassmen only
        if debug:
            logger.info(f'Producing upperclassmen stats for {school_name}')
        
        A_mod = A.copy()
        del A  # Help garbage collection
        gc.collect()
        
        A_mod[list(groups[0]), :] = 0  # Zero out all edges from freshmen
        
        if debug:
            logger.info(f'Producing modified ScaleUp object for {school_name}')
        
        s = ScaleUp(A_mod, groups, [0], directed=True)
        modified = produce_stats(s, debug=debug, sample_size=sample_size)
        for x in modified:
            x['type'] = 'upperclassmen'

        # Free remaining memory
        del s
        del A_mod
        del groups
        gc.collect()

        results = everyone + modified
        
        # Cache results in CSV format
        if debug:
            logger.info(f'Caching results for {school_name}')
        df = pd.DataFrame.from_records(results)
        df.to_csv(csv_file, index=False, sep='\t')
            
        return results
        
    except Exception as e:
        logger.error(f"Error processing {school_name}: {e}")
        return None