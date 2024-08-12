import dataclasses
import functools as ft
import inspect
import re
from collections.abc import Callable
from typing import Any, Generic, Iterable, Optional, overload, ParamSpec, TypeVar, Union

import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PyTree

from ._custom_types import Dims, DimsSpec, sentinel
from ._filters import combine, filter, is_array, partition
from ._module import field, Module, module_update_wrapper, Partial, Static
from ._tree import tree_flatten_one_level


_P = ParamSpec("_P")
_T = TypeVar("_T")

_DIMENSION_NAME = r"\w+"
_DIMENSION_LIST = "^(?:{0:}(?:\\s+{0:})*)?$".format(_DIMENSION_NAME)
_SCALAR = ""


def _is_none(x: Any) -> bool:
    return x is None


def _is_dataclass_or_none(x: Any) -> bool:
    return dataclasses.is_dataclass(x) or _is_none(x)


def _is_vectorized(x: Any, dims: Dims) -> bool:
    return is_array(x) and not _is_none(dims)


_filter = ft.partial(filter, is_leaf=_is_none)
_partition = ft.partial(partition, is_leaf=_is_none)
_tree_flatten = ft.partial(jtu.tree_flatten, is_leaf=_is_none)
_tree_map = ft.partial(jtu.tree_map, is_leaf=_is_none)
_tree_structure = ft.partial(jtu.tree_structure, is_leaf=_is_none)


def _tree_restructure(pytree: PyTree[Any], prototype: PyTree[Any]) -> PyTree[Any]:
    pytree_leaves, _ = _tree_flatten(pytree)
    treedef = _tree_structure(prototype)
    return jtu.tree_unflatten(treedef, pytree_leaves)


def _bind(
    signature: inspect.Signature, args: tuple, kwargs: dict
) -> tuple[tuple, dict]:
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    args = bound.args
    kwargs = bound.kwargs
    return (args, kwargs)


def _combine_input(args: Optional[tuple], kwargs: Optional[dict]) -> Union[dict, tuple]:
    if args and kwargs:
        return (*args, kwargs)
    if args:
        return args
    if kwargs:
        return kwargs
    raise ValueError("Vectorized function must be called with input.")


def _split_input(
    in_: Union[dict, tuple], args: Optional[tuple], kwargs: Optional[dict]
) -> tuple[tuple, dict]:
    if args and kwargs:
        *_args, _kwargs = in_
    elif args:
        _args, _kwargs = in_, {}
    elif kwargs:
        _args, _kwargs = (), in_
    else:
        raise ValueError("Vectorized function must be called with input.")
    return tuple(_args), dict(_kwargs)


def _validate_dims(dims: Dims) -> None:
    if dims is None or re.match(_DIMENSION_LIST, dims):
        return
    raise ValueError(f"'{dims}' is not a valid dimension list.")


def _parse_dims(dims: Dims) -> tuple[str, ...]:
    if dims is None:
        dims = _SCALAR
    _validate_dims(dims)
    return tuple(re.findall(_DIMENSION_NAME, dims))


@dataclasses.dataclass(frozen=True)
class _mangle_dims:
    id: str

    def __call__(self, dims: Dims) -> Dims:
        if dims is None:
            return None
        return " ".join(self.id + "_" + dim for dim in _parse_dims(dims))


@dataclasses.dataclass(frozen=True)
class _batch_dims:
    dims: Dims

    def __call__(self, dims: Dims) -> Dims:
        if self.dims is None or dims is None:
            return None
        return " ".join((*_parse_dims(self.dims), *_parse_dims(dims)))


class DimensionalValue(Module, Generic[_T]):
    value: _T
    dims: DimsSpec = field(static=True)


@dataclasses.dataclass(frozen=True)
class dims_spec:
    dims: Dims = _SCALAR
    mangle: bool = True

    def __call__(self, x: Any) -> PyTree[Dims]:
        if self.dims is None:
            return
        if isinstance(x, Array):
            return self.dims
        if isinstance(x, DimensionalValue):
            return _resolve_dims(x.dims, x.value)
        if not dataclasses.is_dataclass(x):
            return
        dims = []
        for field_ in dataclasses.fields(x):
            if field_.metadata.get("static", False):
                continue
            try:
                value = x.__dict__[field_.name]
            except KeyError:
                continue
            dims_spec_ = field_.metadata.get("dims", _SCALAR)
            dims.append(_resolve_dims(dims_spec_, value))
        if self.mangle:
            id_str = str(id(x))
            dims = _tree_map(_mangle_dims(id_str), dims)
        return _tree_map(_batch_dims(self.dims), dims)


def _resolve_dims_spec(dims_spec_: DimsSpec, pytree: PyTree[Any]) -> PyTree[Dims]:
    if isinstance(dims_spec_, Dims):
        _validate_dims(dims_spec_)
        dims_spec_ = dims_spec(dims_spec_)
    if not callable(dims_spec_):
        raise ValueError("Dimension spec must be a str, None, or Callable[[Any], str].")
    return jtu.tree_map(dims_spec_, pytree, is_leaf=_is_dataclass_or_none)


def _resolve_dims(dims_spec_: PyTree[DimsSpec], pytree: PyTree[Any]) -> PyTree[Dims]:
    resolved_dims = _tree_map(_resolve_dims_spec, dims_spec_, pytree)
    return _tree_restructure(resolved_dims, pytree)


def _gufunc_dims(dims: Dims) -> str:
    return "(" + ",".join(_parse_dims(dims)) + ")"


def _gufunc_dims_list(dims_list: Iterable[Dims]) -> str:
    return ",".join(map(_gufunc_dims, dims_list))


def _gufunc_signature(in_dims: Iterable[Dims], out_dims: Iterable[Dims]) -> str:
    return "->".join(map(_gufunc_dims_list, (in_dims, out_dims)))


class _VectorizeWrapper(Module):
    _fun: Callable
    _in_dims: PyTree[DimsSpec]
    _out_dims: tuple[Dims, ...]
    _signature: inspect.Signature
    _wrap_out: bool

    @property
    def __wrapped__(self):
        return self._fun

    def __call__(self, /, *args, **kwargs):
        # Combine args and kwargs into a single PyTree
        args, kwargs = _bind(self._signature, args, kwargs)
        in_ = _combine_input(args, kwargs)
        # Determine core dimensions of inputs
        in_dims = _resolve_dims(self._in_dims, in_)
        vectorize_filter = _tree_map(_is_vectorized, in_, in_dims)
        # Split input into list of Arrays to be vectorized and excluded PyTree
        vectorize_in, exclude_in = _partition(in_, vectorize_filter)
        vectorize_in_leaves, in_treedef = _tree_flatten(vectorize_in)

        # Function to be wrapped with `jax.numpy.vectorize`
        def _fun_wrapper(*_vectorize_in_leaves):
            # Reconstitute args and kwargs
            _vectorize_in = jtu.tree_unflatten(in_treedef, _vectorize_in_leaves)
            _in_ = combine(_vectorize_in, exclude_in)
            _args, _kwargs = _split_input(_in_, args, kwargs)
            # Call fun with original args and kwargs
            _out = self._fun(*_args, **_kwargs)
            # Wrap output if out_dims was not an Iterable
            if self._wrap_out:
                _out = (_out,)
            # Determine core dimensions of outputs
            _, _out_treedef = tree_flatten_one_level(_out)
            _out_dims_spec = jtu.tree_unflatten(_out_treedef, self._out_dims)
            _out_dims = _resolve_dims(_out_dims_spec, _out)
            # Split output into list of Arrays to be vectorized and excluded PyTree
            _vectorize_filter = _tree_map(_is_vectorized, _out, _out_dims)
            _vectorize_out, _exclude_out = _partition(_out, _vectorize_filter)
            return *_vectorize_out, Static(_exclude_out)

        # Construct gufunc signature
        vectoirze_in_dims = _filter(in_dims, vectorize_filter)
        in_dims_leaves, _ = _tree_flatten(vectoirze_in_dims)
        out_dims_leaves = self._out_dims + (None,)
        gufunc_signature = _gufunc_signature(in_dims_leaves, out_dims_leaves)
        # Call wrapped function
        vectorize_fun = jnp.vectorize(_fun_wrapper, signature=gufunc_signature)
        *vectorize_out, exclude_out = vectorize_fun(*vectorize_in_leaves)
        vectorize_out = _tree_restructure(vectorize_out, exclude_out.value)
        out = combine(vectorize_out, exclude_out.value)
        # Unwrap output if out_dims was not an Iterable
        if self._wrap_out:
            return out[0]
        return out

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return Partial(self, instance)


@overload
def filter_vectorize(
    *,
    in_dims: PyTree[DimsSpec] = _SCALAR,
    out_dims: Union[Dims, Iterable[Dims]] = _SCALAR,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...


@overload
def filter_vectorize(
    fun: Callable[_P, _T],
    *,
    in_dims: PyTree[DimsSpec] = _SCALAR,
    out_dims: Union[Dims, Iterable[Dims]] = _SCALAR,
) -> Callable[_P, _T]: ...


def filter_vectorize(
    fun=sentinel, in_dims=_SCALAR, out_dims: Union[Dims, Iterable[Dims]] = _SCALAR
):
    if fun is sentinel:
        return ft.partial(
            filter_vectorize,
            in_dims=in_dims,
            out_dims=out_dims,
        )
    signature = inspect.signature(fun)
    wrap_out = isinstance(out_dims, Dims)
    if not (wrap_out or isinstance(out_dims, Iterable)):
        raise ValueError(
            "`out_dims` must be a str, None, or an Iterable of str and None."
        )
    vectorize_wrapper = _VectorizeWrapper(
        _fun=fun,
        _in_dims=in_dims,
        _out_dims=(out_dims,) if wrap_out else tuple(out_dims),
        _signature=signature,
        _wrap_out=wrap_out,
    )
    return module_update_wrapper(vectorize_wrapper)
