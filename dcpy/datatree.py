import xarray as xr
from datatree import DataTree, register_datatree_accessor


@register_datatree_accessor("dc")
class MyDataTreeAccessor:
    def __init__(self, tree):
        self._tree = tree

    def extract(self, nodenames):
        return DataTree.from_dict(
            {k: v for k, v in self._tree.children.items() if k in nodenames}
        )

    def subset_nodes(self, varnames) -> DataTree:
        return self._tree.map_over_subtree(lambda n: n[varnames])

    def clear_root(self) -> DataTree:
        new = self._tree.copy()
        for var in self._tree.ds.variables:
            del new.ds[var]
        return new

    def concatenate_nodes(self, dim="node", **concat_kwargs) -> xr.Dataset:
        return xr.concat(
            [
                child.ds.expand_dims({dim: [name]} if dim not in child.ds else dim)
                for name, child in self._tree.children.items()
            ],
            dim=dim,
            **concat_kwargs,
        )

    def update(self, other: DataTree) -> None:
        this = self._tree
        for node in this.children.keys():
            this[node].update(other[node])
