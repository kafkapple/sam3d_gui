# PyTorch 2.0 compatibility for tree_map
from torch.utils import _pytree

def tree_map_compat(fn, *pytrees):
    """tree_map wrapper for PyTorch 2.0 compatibility

    PyTorch 2.0: tree_map(fn, pytree) - only 2 args
    PyTorch 2.1+: tree_map(fn, *pytrees) - multiple pytrees supported
    """
    if len(pytrees) == 1:
        return _pytree.tree_map(fn, pytrees[0])
    elif len(pytrees) == 2:
        # Common case: 2 pytrees - iterate over keys for dicts
        pt1, pt2 = pytrees
        if isinstance(pt1, dict) and isinstance(pt2, dict):
            return {k: fn(pt1[k], pt2[k]) for k in pt1.keys()}
        else:
            # Fallback for non-dict types
            flat1, spec = _pytree.tree_flatten(pt1)
            flat2, _ = _pytree.tree_flatten(pt2)
            results = [fn(a, b) for a, b in zip(flat1, flat2)]
            return _pytree.tree_unflatten(results, spec)
    else:
        # 3+ pytrees
        if all(isinstance(pt, dict) for pt in pytrees):
            keys = pytrees[0].keys()
            return {k: fn(*[pt[k] for pt in pytrees]) for k in keys}
        else:
            flat_lists = [_pytree.tree_flatten(pt)[0] for pt in pytrees]
            spec = _pytree.tree_flatten(pytrees[0])[1]
            results = [fn(*args) for args in zip(*flat_lists)]
            return _pytree.tree_unflatten(results, spec)
