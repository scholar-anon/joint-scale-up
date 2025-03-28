from stats import process_school
import logging
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

# turn on the logging
logging.basicConfig(level=logging.INFO)

school = 'Amherst41'

print('processing school')

start_time = time.time() # start a timer

records = process_school(school, force_recompute=True, debug=True, sample_size=100)

end_time = time.time() # stop the timer
print('time taken:', end_time - start_time)

df_all = pd.DataFrame.from_records(records)

df = df_all[df_all['type'] == 'upperclassmen']
dombrowski = df[df.measure == 'dombrowski']
n_dombrowski = sum( ~np.isnan(dombrowski.value) )
assert n_dombrowski == 0, (
    f'Dombrowski isn\'t possible if our freshmen are not in the samples! '
    f'{n_dombrowski} upperclassmen have Dombrowski estimates:'
    f'[{", ".join(map(str, dombrowski.value[:10]))} ...]'
)

print('There are', len(df_all), 'results')
print('There are', len(df_all[df_all['type'] == 'everyone']), 'results for everyone')
print('There are', len(df_all[df_all['type'] == 'upperclassmen']), 'results for upperclassmen')

from figures import boxplot
boxplot(df)
plt.title('Upperclassmen at Amherst')
plt.show()

df = df_all[df_all['type'] == 'everyone']
assert 'dombrowski' in df.measure.unique(), 'Dombrowski should be in the results for everyone'

boxplot(df)
plt.title('Everyone at Amherst')
plt.show()

print('done')
