import dask
import itertools
import numpy as np
import xarray as xr


def dask_len(obj, optimize=False):
    if optimize:
        return len(dask.optimize(obj.__dask_graph__())[0])
    else:
        return len(obj.__dask_graph__())


def split_blocks(dataset, factor=1):
    """ Splits xarray datasets into its chunks; where each chunk is a dataset

    Inputs
    ------
    dataset: xarray.Dataset
        Dataset to split into blocks
    factor: int
        Multiple of chunksize along each dimension to make one block
    """
    chunk_slices = {}
    for dim, chunks in dataset.chunks.items():
        slices = []
        start = 0
        for chunk in chunks:
            if start >= dataset.sizes[dim]:
                break
            stop = start + factor * chunk
            slices.append(slice(start, stop))
            start = stop
        chunk_slices[dim] = slices
    for slices in itertools.product(*chunk_slices.values()):
        selection = dict(zip(chunk_slices.keys(), slices))
        yield (selection, dataset[selection])


def batch_load(obj, factor=2):
    """
    Load xarray object values by calling compute on block subsets (that are an integral multiple of chunks along each chunked dimension)

    Inputs
    ------

    obj: xarray object
    factor: int
        multiple of chunksize to load at a single time.
        Passed on to split_blocks
    """
    if isinstance(obj, xr.DataArray):
        dataset = obj._to_temp_dataset()
    else:
        dataset = obj

    result = xr.full_like(obj, np.nan).load()
    for label, chunk in split_blocks(dataset, factor=factor):
        print(f"computing {label}")
        computed = chunk.compute()
        print(f"merging...")
        result = xr.merge([result, computed], compat="no_conflicts")

    if isinstance(obj, xr.DataArray):
        obj = obj._from_temp_dataset(result)

    return result
