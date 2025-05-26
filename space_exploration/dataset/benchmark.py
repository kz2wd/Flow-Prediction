
import pandas as pd

def benchmark_dataset(ds_bean):
    df = pd.DataFrame()



    df.to_parquet(
        ds_bean.get_benchmark_storage_name(),
        compression='snappy',
        engine='pyarrow'
    )

