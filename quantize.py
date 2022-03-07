import os
import dask
import dask.dataframe as dd
from dask.distributed import Client
import pandas as pd
import numpy as np
import glob

# method that returns appropriate datatype of each column based on the range of values
def assign_datatype(df):

    data_typ = np.int8
    if (-128 <= df.min_val) and (df.max_val <= 127):
        pass
    elif (-32768 <= df.min_val) and (df.max_val <= 32767):
        data_typ = np.int16
    elif (-2147483648 <= df.min_val) and (df.max_val <= 2147483647):
        data_typ = np.int32
    else:
        data_typ = np.int64
    return data_typ

def get_df_size(ddf):
    series = ddf.memory_usage_per_partition(deep=True).compute()
    return dask.utils.format_bytes(series.sum())


if __name__ == '__main__':

    client = Client()
    files = glob.glob('/home/vineeth/Downloads/archive/Active Wiretap/Active_Wiretap_dataset.csv')
    column_names= ['feature{}'.format(i) for i in range(115)]
    ddf = dd.read_csv(files[0], names=column_names, header=None)
    print("Original Dataframe Size:{}".format(get_df_size(ddf)))
    ddf_int = ddf.astype(np.int32)
    print("Int32 Quantized Dataframe Size:{}".format(get_df_size(ddf_int)))
    mx_vals = ddf.max(axis=0).compute().to_frame(name="max_val")
    min_vals = ddf.min(axis=0).compute().to_frame(name="min_val")
    df_max_mins = pd.concat([min_vals, mx_vals], axis=1)
    df_max_mins['dtype_assignment'] = df_max_mins.apply(assign_datatype, axis=1)
    dtype_assignment = df_max_mins['dtype_assignment'].to_dict()
    ddf = ddf.astype(dtype_assignment)
    print("Value-Specific Quantized Dataframe Size:{}".format(get_df_size(ddf)))
    client.close()