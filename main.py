import pandas as pd

df = pd.read_pickle('data/df.pickle')

# df = pd.read_csv('data/source.csv')

pd.set_option('display.max_columns', None)
print(df.dtypes)

