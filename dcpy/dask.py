import itertools

import dask
import numpy as np

import xarray as xr


def ntasks(obj, optimize=False):
    """Returns length of dask graph.

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


def visualize_one_block(dataset, **kwargs):
    """
    Visualize one block of a Dataset or DataArray.
    """
    graph = None
    if isinstance(dataset, xr.DataArray):
        dataset = dataset._to_temp_dataset()

    keys = []
    block = get_one_block(dataset.unify_chunks())
    graph = block.__dask_graph__()

    for name, variable in block.variables.items():
        if isinstance(variable.data, dask.array.Array):
            key = (variable.data.name,) + (0,) * variable.ndim
            keys.append(key)

    if graph is None:
        raise ValueError("No dask variables!")
    return dask.visualize(graph.cull(set(keys)), **kwargs)


def get_one_block(dataset):
    if isinstance(dataset, xr.DataArray):
        array = dataset
        dataset = dataset._to_temp_dataset()
        to_array = True
    else:
        to_array = False
    chunk_slices = dict()
    for dim, chunks in dataset.chunks.items():
        start = 0
        chunk = chunks[0]
        stop = start + chunk
        chunk_slices[dim] = slice(start, stop)
    subset = dataset.isel(chunk_slices)
    if to_array:
        subset = array._from_temp_dataset(subset)
    return subset


def split_blocks(dataset, factor=1):
    """Splits xarray datasets into its chunks; where each chunk is a dataset

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


def batch_to_zarr(ds, file, dim, batch_size, restart=False, **kwargs):
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

    if not restart:
        ds.isel({dim: [0]}).to_zarr(file, consolidated=True, mode="w", **kwargs)
    else:
        print("Restarting...")
        opened = xr.open_zarr(file, consolidated=True)
        ds = ds.sel(time=slice(opened[dim][-1], None))
        print(
            f"Last index = {opened[dim][-1].values}. Starting from {ds[dim][1].values}"
        )
        opened.close()

    for t in tqdm.tqdm(range(1, ds.sizes[dim], batch_size)):
        if "encoding" in kwargs:
            del kwargs["encoding"]
        ds.isel({dim: slice(t, t + batch_size)}).to_zarr(
            file, consolidated=True, mode="a", append_dim=dim, **kwargs
        )


def map_copy(obj):
    from xarray import DataArray
    import numpy as np

    if isinstance(obj, DataArray):
        name = obj.name
        dataset = obj._to_temp_dataset()
    else:
        dataset = obj.copy(deep=True)

    for var in dataset.variables:
        if dask.is_dask_collection(dataset[var]):
            dataset[var].data = dataset[var].data.map_blocks(np.copy)

    if isinstance(obj, DataArray):
        result = obj._from_temp_dataset(dataset)
    else:
        result = dataset

    return result


#
# Examples of the 'fetch all the chunks' dask memory problem,
# and some possible solutions :
#    (1) a "safeslice" operation, which constructs a lazy calculation that
#        slices + each sourcedata chunk separately
#    (2) a "compute_via_store" operation, which uses "da.store" to save the result into
#        a pre-allocated result array.
#


# Control for alternative code approach.
# N.B. seems **equivalent**, since we added the "map_blocks(np.copy)" step ...
EMBED_INDEXES = False


def dask_safeslice(data, indices, chunks=None):
    """
    COPIED FROM https://github.com/dask/dask/issues/5540#issuecomment-601150129
    Added fancy indexing xarray.core.indexing.DaskIndexingAdapter

    Return a subset of a dask array, but with indexing applied independently to
    each slice of the input array, *prior* to their recombination to produce
    the result array.

    Args:

    * data (dask array):
        input data
    * indices (int or slice or tuple(int or slice)):
        required sub-section of the data.

    Kwargs:

    * chunks (list of (int or "auto")):
        chunking argument for 'rechunk' applied to the input.
        If set, forces the input to be rechunked as specified.
        ( This replaces the normal operation, which is to rechunk the input
        making the indexed dimensions undivided ).
        Mainly for testing on small arrays.

    .. note::

        'indices' currently does not support Ellipsis or newaxis.

    """

    from collections.abc import Iterable
    import dask.array as da

    # The idea is to "push down" the indexing operation to "underneath" the
    # result concatenation, so it gets done _before_ that.
    # This 'result concatenation' is actually implicit: the _implied_
    # concatenation of all the result chunks into a single output array.
    # We assume that any *one* chunk *can* be successfully computed.
    # By applying the indexing operation to each chunk, prior to the
    # complete result (re-)construction, we hope to make this work.

    # Normalise input to a list over all data dimensions.

    # NOTE: FOR NOW, this does not support Ellipsis.
    # TODO: that could easily be fixed.

    # Convert the slicing indices to a list of (int or slice).
    # ( NOTE: not supporting Ellipsis. )
    if not isinstance(indices, Iterable):
        # Convert a single key (slice or integer) to a length-1 list.
        indices = [indices]
    else:
        # Convert other iterable types to lists.
        indices = list(indices)

    n_data_dims = data.ndim
    assert len(indices) <= n_data_dims

    # Extend with ":" in all the additional (trailing) dims.
    all_slice = slice(None)
    indices += (n_data_dims - len(indices)) * [all_slice]

    assert len(indices) == n_data_dims

    # Discriminate indexed and non-indexed dims.
    # An "indexed" dim is where input index is *anything* other than a ":".
    dim_is_indexed = [index != all_slice for index in indices]

    # Work out which indices are simple integer values.
    # ( by definition, all of these will be "indexed" dims )
    dim_is_removed = [isinstance(key, int) for key in indices]

    # Replace single-value indices with length-1 indices, so the indexing
    # preserves all dimensions (as this makes reconstruction easier).
    # ( We use the above 'dim_is_removed' to correct this afterwards. )
    indices = [slice(key, key + 1) if isinstance(key, int) else key for key in indices]

    # We will now rechunk to get "our chunks" : but these must not be divided
    # in dimensions affected by the requested indexing.
    # So we rechunk, but insist that those dimensions are kept whole.
    # ( Obviously, not always optimal ... )
    # As the indexed dimensions will always be _reduced_ by the indexing, this
    # is obviously over-conservative + may give chunks which are rather too
    # small.  Let's just ignore that problem for now!
    if chunks is not None:
        rechunk_dim_specs = list(chunks)
    else:
        rechunk_dim_specs = ["auto"] * n_data_dims
    for i_dim in range(n_data_dims):
        if dim_is_indexed[i_dim]:
            rechunk_dim_specs[i_dim] = -1
    data = da.rechunk(data, chunks=rechunk_dim_specs)

    # Calculate multidimensional indexings of the original data array which
    # correspond to all these chunks.
    # Note: following the "-1"s in the above rechunking spec, the indexed dims
    # should all have only one chunk in them.
    assert all(
        len(data.chunks[i_dim]) == 1
        for i_dim in range(n_data_dims)
        if dim_is_removed[i_dim]
    )

    # Make an array of multidimensional indexes corresponding to all chunks.
    chunks_shape = [len(chunk_lengths) for chunk_lengths in data.chunks]
    chunks_shape += [n_data_dims]
    chunk_indices = np.zeros(chunks_shape, dtype=object)
    # The chunk_indices array ...
    #     * has dimensions of n-data-dims + 1
    #     * has shape of "chunks-shape" + (n_data_dims,)
    #     * each entry[i0, i1, iN-1] --> n_data_dims * slice-objects.

    # Pre-fill indexes array with [:, :, ...]
    chunk_indices[...] = all_slice
    # Set slice ranges for each dimension at a time.
    for i_dim in range(n_data_dims):
        # Fix all keys for this data dimension : chunk_indices[..., i_dim]
        dim_inds = [all_slice] * n_data_dims + [i_dim]
        if dim_is_indexed[i_dim]:
            # This is a user-indexed dim, so should be un-chunked.
            assert len(data.chunks[i_dim]) == 1
            # Set keys for this dim to the user-requested indexing.
            if EMBED_INDEXES:
                chunk_indices[tuple(dim_inds)] = indices[i_dim]
        else:
            # Replace keys for this dim with the slice range for the
            # relevant chunk, for each chunk in the dim.
            startend_positions = np.cumsum([0] + list(data.chunks[i_dim]))
            starts, ends = startend_positions[:-1], startend_positions[1:]
            for i_key, (i_start, i_end) in enumerate(zip(starts, ends)):
                dim_inds[i_dim] = i_key
                chunk_indices[tuple(dim_inds)] = slice(i_start, i_end)
                # E.G. chunk_indices[:, :, 1, :][2] = slice(3,6)

    # Make actual addressed chunks by indexing the original array, arrange them
    # in the same pattern, and re-combine them all to make a result array.
    # This needs to be a list-of-lists construction, as da.block requires it.
    # ( an array of arrays is presumably too confusing ?!? )
    def get_chunks(multidim_indices):
        if multidim_indices.ndim > 1:
            # Convert the "array of chunks" dims --> lists-of-lists
            result = [
                get_chunks(multidim_indices[i_part])
                for i_part in range(multidim_indices.shape[0])
            ]
        else:
            # Innermost dim contains n-dims * slice-objects
            # Convert these into a slice of the data array.
            result = data.__getitem__(tuple(multidim_indices))

            if not EMBED_INDEXES:
                # Now *also* apply the required indexing to this chunk.
                # It initially seemed *essential* that this be an independent
                # operation, so that the memory associated with the whole chunk
                # can be released.
                # But ACTUALLY this is not so, given the next step (see on).
                try:
                    result = result.__getitem__(tuple(indices))
                except NotImplementedError:
                    result = data
                    for axis, subkey in reversed(list(enumerate(tuple(indices)))):
                        result = result[(slice(None),) * axis + (subkey,)]

            # AND FINALLY : apply a numpy copy to this indexed-chunk.
            # This is essential, to release the source chunks ??
            # see: https://github.com/dask/dask/issues/3595#issuecomment-449546228
            result = result.map_blocks(np.copy)

        return result

    listoflists_of_chunks = get_chunks(chunk_indices)
    result = da.block(listoflists_of_chunks)

    assert result.ndim == n_data_dims  # Unchanged as 'da.block' concatenates.

    # Finally remove the extra dimensions for single-value indices.
    assert all(
        result.shape[i_dim] == 1
        for i_dim in range(n_data_dims)
        if dim_is_removed[i_dim]
    )
    all_dim_indices = [
        0 if dim_is_removed[i_dim] else all_slice for i_dim in range(n_data_dims)
    ]
    result = result.__getitem__(tuple(all_dim_indices))
    return result


def index(da, indexers):
    from xarray.core.indexing import remap_label_indexers

    if not isinstance(da, xr.DataArray):
        raise TypeError(f"Expected DataArray. Received {type(da).__name__}")
    pos_indexers, new_indexes = remap_label_indexers(da, indexers)
    dask_indexers = list(pos_indexers.values())

    # TODO: avoid the sel. That could be slow
    indexed = da.sel(**indexers).copy(data=dask_safeslice(da.data, dask_indexers))
    return indexed
