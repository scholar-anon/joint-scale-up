from pathlib import Path
import scipy.io
import pandas as pd

from params import *

files = Path(base_dir).rglob("*.mat")
files = list(files)
school_names = [f.stem for f in files]

def load_school(school_name):
    m = scipy.io.loadmat(str(Path(base_dir) / f"{school_name}.mat"))
    A = m["A"]
    df = make_df(m)

    A = A[df.index, :][:, df.index] # respect the filtering
    
    # reindex the dataframe for consistency
    df.index = range(len(df))
    df = df.reset_index(drop=True)
    
    return A, df

def make_df(mat):
    if "local_info" not in mat:
        return None
    
    info = pd.DataFrame(
        mat["local_info"],
        columns=[
            "student/faculty",
            "gender",
            "major",
            "second major/minor",
            "dorm/house",
            "year",
            "high school",
        ]
    )

    # missing data is coded zero
    info.replace(0, pd.NA)

    # filter to only include students
    info = info[info["student/faculty"] == 1]

    return info

def load_results(fn):
  import os
  import pickle
  import pandas as pd
  
  # Check if cached results exist
  stats_dir.mkdir(parents=True, exist_ok=True)
  csv_file = stats_dir / f'{fn}.csv'
  pickle_file = stats_dir / f'{fn}.pkl'
  
  # Try CSV first
  if os.path.exists(csv_file):
    df = pd.read_csv(csv_file, sep='\t')
    rows = df.to_dict('records')
    for r in rows:
      r['school'] = fn.split('_stats.csv')[0]
    return rows
    
  # Fall back to pickle if CSV doesn't exist
  if os.path.exists(pickle_file):
    with pickle_file.open('rb') as f:
      rows = pickle.load(f)
      for r in rows:
        r['school'] = fn.split('_stats.pkl')[0]
      return rows
    
  return None

def load_all_results():
    results = []
    for s in school_names:
        mine = load_results(s)
        if mine is None: continue

        mine = [r for r in mine if r is not None]
        for r in mine:
            r['school'] = s

        results.extend(mine)

    df_all = pd.DataFrame.from_records(results)
    return df_all
