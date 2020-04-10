import pandas as pd
import numpy as np
import math
import scipy.stats as si
PATH = 'https://raw.githubusercontent.com/hmku/mwtc-2020/master/MWTC-platform-release-1.1.1/data/normalized_price_paths/history_0.csv'

df = pd.read_csv(PATH, index_col=0, header=0).transpose()
df.pct_change().std()