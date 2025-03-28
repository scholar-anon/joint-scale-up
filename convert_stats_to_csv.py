"""
Convert pickle stats files to CSV format for better compatibility.
"""

import os
import pickle
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from params import stats_dir

def convert_pickle_to_csv(pickle_file: Path) -> None:
    """
    Convert a single pickle stats file to CSV format.
    
    Args:
        pickle_file: Path to the pickle file to convert
    """
    # Read pickle file
    with pickle_file.open('rb') as f:
        rows = pickle.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame.from_records(rows)
    
    # Add school name if not present
    if 'school' not in df.columns:
        df['school'] = pickle_file.stem
    
    # Save as CSV
    csv_file = pickle_file.with_suffix('.csv')
    df.to_csv(csv_file, index=False, sep='\t')

def main():
    """Convert all pickle stats files to CSV format."""
    # Create stats directory if it doesn't exist
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all pickle files
    pickle_files = list(stats_dir.glob('*.pkl'))
    
    print(f'Found {len(pickle_files)} pickle files to convert')
    
    # Convert each file
    for pickle_file in tqdm(pickle_files, desc='Converting files'):
        try:
            convert_pickle_to_csv(pickle_file)
        except Exception as e:
            print(f'Error converting {pickle_file}: {e}')

if __name__ == '__main__':
    main() 