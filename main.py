import pandas as pd

df = pd.read_pickle('data/df.pickle')

pd.set_option('display.max_columns', None)
print(df.dtypes)