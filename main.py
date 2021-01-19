import pandas as pd

# df = pd.read_pickle('data/df.pickle')

df = pd.read_csv('data/source.csv')

pd.set_option('display.max_columns', None)
# print(df.dtypes)


def reduce_precision(df):
    import numpy as np
    """
    usage: give a dataframe, this fuction returns an optimized dataframe

    df = reduce_precision(df)

    reference: https://gist.github.com/enamoria/fa9baa906f23d1636c002e7186516a7b
    """
    cols_to_convert = []
    date_strings = ['_date', 'date_', 'date']

    for col in df.columns:
        col_type = df[col].dtype
        if 'string' not in col_type.name and col_type.name != 'category' and 'datetime' not in col_type.name:
            cols_to_convert.append(col)

    def _reduce_precision(x):
        col_type = x.dtype
        unique_data = list(x.unique())
        bools = [True, False, 'true', 'True', 'False', 'false']
        n_unique = float(len(unique_data))
        n_records = float(len(x))
        cat_ratio = n_unique / n_records

        try:
            unique_data.remove(np.nan)
        except:
            pass

        if 'int' in str(col_type):
            c_min = x.min()
            c_max = x.max()

            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                x = x.astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                x = x.astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                x = x.astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                x = x.astype(np.int64)

                # TODO: set precision to unsigned integers with nullable NA

        elif 'float' in str(col_type):
            c_min = x.min()
            c_max = x.max()
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                x = x.astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                x = x.astype(np.float32)
            else:
                x = x.astype(np.float64)

        elif 'datetime' in col_type.name or any(i in str(x.name).lower() for i in date_strings):
            try:
                x = pd.to_datetime(x)
            except:
                pass

        elif any(i in bools for i in unique_data):
            x = x.astype('boolean')
            # TODO: set precision to bool if boolean not needed

        elif cat_ratio < .1 or n_unique < 20:
            x = x.astype('category')

        elif all(isinstance(i, str) for i in unique_data):
            x = x.astype('string')

        return x

    df[cols_to_convert] = df[cols_to_convert].apply(lambda x: _reduce_precision(x))

    return df

df = reduce_precision(df)

# print(df.dtypes)

