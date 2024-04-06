import dataclasses
import functools as ft
import re
from collections.abc import Callable
from typing import Any, Optional, overload, ParamSpec, TypeVar

import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PyTree

from ._custom_types import Dims, DimsSpec, sentinel
from ._filters import combine, filter, is_array, partition
from ._module import Module, module_update_wrapper, Partial


_P = ParamSpec("_P")
_T = TypeVar("_T")

_DIMENSION_NAME = r"\w+"
_DIMENSION_LIST = "(?:{0:}(?:,{0:})*)?".format(_DIMENSION_NAME)
_ARGUMENT = rf"\({_DIMENSION_LIST}\)"
_SCALAR = "()"


def _is_none(x: Any) -> bool:
    return x is None


_tree_map = ft.partial(jtu.tree_map, is_leaf=_is_none)
_tree_flatten = ft.partial(jtu.tree_flatten, is_leaf=_is_none)
_filter = ft.partial(filter, is_leaf=_is_none)


def _is_dataclass_or_none(x: Any) -> bool:
    return dataclasses.is_dataclass(x) or x is None


def _is_exclude(x: Any, dims: Optional[Dims]) -> bool:
    return not is_array(x) or dims is None


def _parse_dims(dims: str) -> tuple[str, ...]:
    if not re.match(_ARGUMENT, dims):
        raise ValueError(f"{dims} is not a valid gufunc dimension signature.")
    return tuple(re.findall(_DIMENSION_NAME, dims))


@dataclasses.dataclass(frozen=True)
class _mangle_dims:
    id: str

    def __call__(self, dims: Dims) -> Dims:
        if dims is None:
            return None
        return "(" + ",".join(self.id + "_" + dim for dim in _parse_dims(dims)) + ")"


@dataclasses.dataclass(frozen=True)
class _batch_dims:
    batch: Dims

    def __call__(self, dims: Dims) -> Dims:
        if self.batch is None or dims is None:
            return None
        return "(" + ",".join((*_parse_dims(self.batch), *_parse_dims(dims))) + ")"


@dataclasses.dataclass(frozen=True)
class dims_spec:
    dims: Dims = _SCALAR
    exclude: bool = False
    mangle: bool = True

    def __call__(self, x: Any) -> PyTree[Dims]:
        if self.exclude:
            return
        if isinstance(x, Array):
            return self.dims
        if dataclasses.is_dataclass(x):
            dims = []
            for field in dataclasses.fields(x):
                if field.metadata.get("static", False):
                    continue
                try:
                    value = x.__dict__[field.name]
                except KeyError:
                    continue
                dims_spec = field.metadata.get("dims", self.__class__())
                resolved_dims = _resolve_dims(value, dims_spec)
                dims.append(resolved_dims)
            if self.mangle:
                dims = _tree_map(_mangle_dims(str(id(x))), dims)
            return _tree_map(_batch_dims(self.dims), dims)


def _resolve_dims_spec(_dims_spec: DimsSpec, elem: Any) -> PyTree[Dims]:
    if _dims_spec is None:
        _dims_spec = dims_spec(exclude=True)
    if isinstance(_dims_spec, Dims):
        _dims_spec = dims_spec(dims=_dims_spec)
    if not callable(_dims_spec):
        raise ValueError("`in_dims` must be a PyTree of strings and callables only.")
    return jtu.tree_map(_dims_spec, elem, is_leaf=_is_dataclass_or_none)


def _resolve_dims(pytree: PyTree[Any], dims_spec: PyTree[DimsSpec]) -> PyTree[Dims]:
    return _tree_map(_resolve_dims_spec, dims_spec, pytree)


def _combine_args_kwargs(args, kwargs):
    if args and kwargs:
        return (*args, kwargs)
    if args:
        return args
    if kwargs:
        return kwargs
    raise ValueError("Vectorized function must be called with input.")


def _split_in(args, kwargs, in_):
    if args and kwargs:
        *_args, _kwargs = in_
    elif args:
        _args, _kwargs = in_, {}
    elif kwargs:
        _args, _kwargs = (), in_
    else:
        raise ValueError("Vectorized function must be called with input.")
    return _args, _kwargs


def _signature(
    in_dims_leaves: PyTree[Dims], filter_leaves: PyTree[bool], out_dims: str
) -> str:
    in_dims_leaves = _filter(in_dims_leaves, filter_leaves, True, _SCALAR)
    in_dims = ",".join(in_dims_leaves)
    signature = "->".join((in_dims, out_dims))
    return signature


class _VectorizeWrapper(Module):
    _fun: Callable
    _in_dims: PyTree[DimsSpec]
    _out_dims: str

    @property
    def __wrapped__(self):
        return self._fun

    def __call__(self, /, *args, **kwargs):
        in_ = _combine_args_kwargs(args, kwargs)
        in_dims = _resolve_dims(in_, self._in_dims)
        in_leaves, in_treedef = _tree_flatten(in_)
        in_dims_leaves, _ = _tree_flatten(in_dims)
        filter_leaves = _tree_map(_is_exclude, in_leaves, in_dims_leaves)
        exclude_in_leaves, include_in_leaves = partition(in_leaves, filter_leaves)
        dynamic_in_leaves, static_in_leaves = partition(include_in_leaves, is_array)
        static_in_leaves = tuple(combine(exclude_in_leaves, static_in_leaves))

        def _fun_wrapper(*_dynamic_in_leaves):
            _in_leaves = combine(_dynamic_in_leaves, static_in_leaves)
            _in_ = jtu.tree_unflatten(in_treedef, _in_leaves)
            _args, _kwargs = _split_in(args, kwargs, _in_)
            return self._fun(*_args, **_kwargs)

        signature = _signature(in_dims_leaves, filter_leaves, self._out_dims)
        return jnp.vectorize(_fun_wrapper, signature=signature)(*dynamic_in_leaves)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return Partial(self, instance)


@overload
def filter_vectorize(
    *, in_dims: PyTree[DimsSpec] = _SCALAR, out_dims: str = _SCALAR
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    ...


@overload
def filter_vectorize(
    fun: Callable[_P, _T],
    *,
    in_dims: PyTree[DimsSpec] = _SCALAR,
    out_dims: str = _SCALAR,
) -> Callable[_P, _T]:
    ...


def filter_vectorize(fun=sentinel, in_dims=_SCALAR, out_dims=_SCALAR):
    if fun is sentinel:
        return ft.partial(
            filter_vectorize,
            in_dims=in_dims,
            out_dims=out_dims,
        )

    vectorize_wrapper = _VectorizeWrapper(
        _fun=fun,
        _in_dims=in_dims,
        _out_dims=out_dims,
    )
    return module_update_wrapper(vectorize_wrapper)
