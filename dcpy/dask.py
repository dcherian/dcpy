import dask
import itertools
import numpy as np
import xarray as xr


def ntasks(obj, optimize=False):
    """ Returns length of dask graph.

    Parameters
    ----------
    optimize: bool, optional
        Optimize graph?

    Returns
    -------
    number of tasks, int
    """
    if optimize:
        return len(dask.optimize(obj)[0].__dask_graph__())
    else:
        return len(obj.__dask_graph__())


def split_blocks(dataset, factor=1):
    """ Splits xarray datasets into its chunks; where each chunk is a dataset

    Parameters
    ----------
    dataset: xarray.Dataset
        Dataset to split into blocks
    factor: int
        Multiple of chunksize along each dimension to make one block

    Returns
    -------
    generator yielding ("isel dictionary", dataset)
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

    Parameters
    ----------
    obj: xarray object
    factor: int
        multiple of chunksize to load at a single time.
        Passed on to split_blocks
    """
    if isinstance(obj, xr.DataArray):
        dataset = obj._to_temp_dataset()
    else:
        dataset = obj

    # result = xr.full_like(obj, np.nan).load()
    computed = []
    for label, chunk in split_blocks(dataset, factor=factor):
        print(f"computing {label}")
        computed.append(chunk.compute())
    result = xr.combine_by_coords(computed)

    if isinstance(obj, xr.DataArray):
        result = obj._from_temp_dataset(result)

    return result


def batch_to_zarr(ds, file, dim, batch_size, **kwargs):
    """
    Batched writing of dask arrays to zarr files.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to write.
    file : str
        filename
    dim : str
        Dimension along which to split dataset and append. Passed to `to_zarr` as `append_dim`
    batch_size : int
        Size of a single batch

    Returns
    -------
    None
    """

    import tqdm

    ds.isel({dim: [0]}).to_zarr(file, consolidated=True, mode="w", **kwargs)

    for t in tqdm.tqdm(range(1, ds.sizes[dim], batch_size)):
        if "encoding" in kwargs:
            del kwargs["encoding"]
        ds.isel({dim: slice(t, t + batch_size)}).to_zarr(
            file, consolidated=True, mode="a", append_dim=dim, **kwargs
        )
