import multiprocessing
import random
import time
from typing import List, Dict
from tqdm import tqdm
import logging

from stats import process_school, is_memory_stable
from load_data import school_names

logger = logging.getLogger(__name__)

def process_schools(schools: List[str], process_school_func, kwargs: Dict = {}) -> List[Dict]:
    """
    Process multiple schools in parallel with memory management.
    
    Args:
        schools: List of school names to process
        process_school_func: Function to process each school
        
    Returns:
        List of dictionaries containing all results
    """
    rows = []
    remaining_schools = list(schools)

    # shuffle schools, I can't figure out how to order them intelligently
    random.shuffle(remaining_schools)

    active_processes = {}

    first = True
    
    pbar = tqdm(total=len(schools), desc="Processing schools")
    
    while remaining_schools or active_processes:
        # Start new processes if memory is consistently low
        while remaining_schools:
            school = remaining_schools[0]
            
            # Check cache first
            if not kwargs.get('force_recompute', False):
                possible_results = process_school_func(school, cache_only=True)
                if possible_results:
                    pbar.set_description(f"Skipping {school}")
                    rows.extend(possible_results)
                    pbar.update(1)
                    remaining_schools.pop(0)
                    continue
            
            n_running = len(active_processes)

            # Check if memory is stablely above 50%
            if not first:
                stable, mem = is_memory_stable(duration=60, checks=3)
                if not stable:
                    if mem > 90:
                        # if memory is above 90%, that's too much and will cause problems
                        # kill the most recently started process, and move that school to the end of the list
                        process = list(active_processes.keys())[-1]
                        school = active_processes[process]
                        # kill the process
                        process.terminate()
                        process.join()
                        # move the school to the end of the list
                        remaining_schools.append(school)
                        # decrement the pbar
                        pbar.update(-1)
                        # update status
                        pbar.set_description(f"Killed {school} due to memory issues")
                        break

                    pbar.set_description(f"Waiting for memory... ({n_running} running, {mem}%)")
                    break

            first = False

            school = remaining_schools.pop(0)
                
            pbar.set_description(f"Processing {school} ({n_running+1} running)")
            
            # Start new process
            process = multiprocessing.Process(
                target=process_school_func,
                args=(school,),
                kwargs=kwargs
            )
            process.start()
            active_processes[process] = school
            
        # Check completed processes
        completed = []
        for process, school in active_processes.items():
            if not process.is_alive():
                completed.append(process)
                process.join()
                
                try:
                    # Load results from cache since process is done
                    results = process_school_func(school, cache_only=True)
                    if results:
                        rows.extend(results)
                    pbar.update(1)
                except Exception as exc:
                    logger.error(f'{school} generated an exception: {exc}')
                    pbar.update(1)
                    
        # Cleanup completed processes
        for process in completed:
            del active_processes[process]
            
        # Brief pause before next check if work remains
        if remaining_schools or active_processes:
            time.sleep(10)
            
    pbar.close()
    return rows

if __name__ == '__main__':
    # only process the schools we already have - which we know will work
    # there's one that just takes up too much memory to run, and that's fine
    from params import stats_dir
    school_names = [
        name for name in school_names
        if (stats_dir.with_name('stats copy') / f'{name}.pkl').exists()
    ]

    print(f'Processing {len(school_names)} schools')

    results = process_schools(school_names, process_school, {
        'debug': False, 'sample_size': 250, 'force_recompute': True
    })