import xarray as xr
from datatree import DataTree, register_datatree_accessor


@register_datatree_accessor("dc")
class MyDataTreeAccessor:
    def __init__(self, tree):
        self._tree = tree

    def extract_leaf(self, leaf):
        return DataTree.from_dict(
            {node.parent.name: node for node in self._tree.leaves if node.name == leaf}
        )

    def reorder_nodes(self, order):
        tree = self._tree
        if ... in order:
            allnames = tuple(tree.children.keys())
            remaining = [v for v in allnames if v not in order]

        neworder = []
        for k in order:
            if k != ...:
                neworder.append(k)
            else:
                neworder.extend(remaining)
        newtree = DataTree.from_dict({k: tree[k] for k in neworder})
        return newtree

    def extract(self, nodenames):
        return DataTree.from_dict(
            {k: v for k, v in self._tree.children.items() if k in nodenames}
        )

    def subset_nodes(self, varnames) -> DataTree:
        return self._tree.map_over_subtree(lambda n: n.cf[varnames])

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

    def insert_as_subtree(self, name, other) -> DataTree:
        newtree = DataTree()
        for (ourkey, ours), (theirkey, theirs) in zip(
            self._tree.children.items(), other.children.items()
        ):
            assert ourkey == theirkey
            newtree[ourkey] = ours
            newtree[ourkey][name] = theirs
        return newtree
