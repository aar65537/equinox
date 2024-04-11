import dataclasses
import functools as ft
import inspect
import re
from collections.abc import Callable
from typing import Any, Iterable, Optional, overload, ParamSpec, TypeVar, Union

import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PyTree

from ._custom_types import Dims, DimsSpec, sentinel
from ._filters import combine, filter, is_array, partition
from ._module import Module, module_update_wrapper, Partial, Static


_P = ParamSpec("_P")
_T = TypeVar("_T")

_DIMENSION_NAME = r"\w+"
_DIMENSION_LIST = "^(?:{0:}(?:\\s+{0:})*)?$".format(_DIMENSION_NAME)
_SCALAR = ""


def _is_none(x: Any) -> bool:
    return x is None


_filter = ft.partial(filter, is_leaf=_is_none)
_tree_flatten = ft.partial(jtu.tree_flatten, is_leaf=_is_none)
_tree_map = ft.partial(jtu.tree_map, is_leaf=_is_none)
_tree_structure = ft.partial(jtu.tree_structure, is_leaf=_is_none)


def _tree_restructure(pytree: PyTree, prototype: PyTree) -> PyTree:
    pytree_leaves, _ = _tree_flatten(pytree)
    treedef = _tree_structure(prototype)
    return jtu.tree_unflatten(treedef, pytree_leaves)


def _is_dataclass_or_none(x: Any) -> bool:
    return dataclasses.is_dataclass(x) or x is None


def _is_exclude(x: Any, dims: Optional[Dims]) -> bool:
    return not is_array(x) or dims is None


def _parse_dims(dims: str) -> tuple[str, ...]:
    if not re.match(_DIMENSION_LIST, dims):
        raise ValueError(f"'{dims}' is not a valid dimension signature.")
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
    batch: Dims

    def __call__(self, dims: Dims) -> Dims:
        if self.batch is None or dims is None:
            return None
        return " ".join((*_parse_dims(self.batch), *_parse_dims(dims)))


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
                id_str = str(id(x))
                dims = _tree_map(_mangle_dims(id_str), dims)
            return _tree_map(_batch_dims(self.dims), dims)


def _resolve_dims_spec(_dims_spec: DimsSpec, elem: Any) -> PyTree[Dims]:
    if _dims_spec is None:
        _dims_spec = dims_spec(exclude=True)
    elif isinstance(_dims_spec, Dims):
        _dims_spec = dims_spec(dims=_dims_spec)
    if not callable(_dims_spec):
        raise ValueError("`in_dims` must be a PyTree of strings and callables only.")
    return jtu.tree_map(_dims_spec, elem, is_leaf=_is_dataclass_or_none)


def _resolve_dims(pytree: PyTree[Any], dims_spec: PyTree[DimsSpec]) -> PyTree[Dims]:
    return _tree_map(_resolve_dims_spec, dims_spec, pytree)


def _combine_args_kwargs(
    args: Optional[tuple], kwargs: Optional[dict]
) -> Union[dict, tuple]:
    if args and kwargs:
        return (*args, kwargs)
    if args:
        return args
    if kwargs:
        return kwargs
    raise ValueError("Vectorized function must be called with input.")


def _split_in(
    args: Optional[tuple], kwargs: Optional[dict], in_: Union[dict, tuple]
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


def _convert_dims(dims: str) -> str:
    return "(" + ",".join(dim for dim in _parse_dims(dims)) + ")"


def _convert_to_signature(dims_leaves: Iterable[str]) -> str:
    return ",".join(map(_convert_dims, dims_leaves))


def _gufunc_signature(
    in_dims_leaves: PyTree[Dims],
    filter_leaves: PyTree[bool],
    out_dims: tuple[Dims, ...],
) -> str:
    in_dims_leaves = _filter(in_dims_leaves, filter_leaves, True, _SCALAR)
    out_dims_leaves = (_SCALAR if dim is None else dim for dim in out_dims + (None,))
    in_signature = _convert_to_signature(in_dims_leaves)
    out_signature = _convert_to_signature(out_dims_leaves)
    return in_signature + "->" + out_signature


def _bind(
    signature: inspect.Signature, args: tuple, kwargs: dict
) -> tuple[tuple, dict]:
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    args = bound.args
    kwargs = bound.kwargs
    return (args, kwargs)


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
        args, kwargs = _bind(self._signature, args, kwargs)
        in_ = _combine_args_kwargs(args, kwargs)
        in_dims = _resolve_dims(in_, self._in_dims)
        in_leaves, in_treedef = _tree_flatten(in_)
        in_dims_leaves, _ = _tree_flatten(in_dims)
        filter_leaves = _tree_map(_is_exclude, in_leaves, in_dims_leaves)
        exclude_in_leaves, include_in_leaves = partition(in_leaves, filter_leaves)
        dynamic_in_leaves, static_in_leaves = partition(include_in_leaves, is_array)
        static_in_leaves = tuple(combine(exclude_in_leaves, static_in_leaves))
        gufunc_signature = _gufunc_signature(
            in_dims_leaves, filter_leaves, self._out_dims
        )

        def _fun_wrapper(*_dynamic_in_leaves):
            _in_leaves = combine(_dynamic_in_leaves, static_in_leaves)
            _in_ = jtu.tree_unflatten(in_treedef, _in_leaves)
            _args, _kwargs = _split_in(args, kwargs, _in_)
            _out = self._fun(*_args, **_kwargs)
            _out = (_out,) if self._wrap_out else _out
            _out_dims = _resolve_dims(tuple(_out), self._out_dims)
            _out_dims = _tree_restructure(_out_dims, _out)
            _filter = _tree_map(_is_exclude, _out, _out_dims)
            _exclude_out, _include_out = partition(_out, _filter)
            return *_include_out, Static(_exclude_out)

        vectorized_fun = jnp.vectorize(_fun_wrapper, signature=gufunc_signature)
        *include_out, exclude_out = vectorized_fun(*dynamic_in_leaves)
        include_out = _tree_restructure(include_out, exclude_out.value)
        out = combine(include_out, exclude_out.value)
        return out[0] if self._wrap_out else out

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return Partial(self, instance)


@overload
def filter_vectorize(
    *,
    in_dims: PyTree[DimsSpec] = _SCALAR,
    out_dims: Union[Dims, Iterable[Dims]] = _SCALAR,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    ...


@overload
def filter_vectorize(
    fun: Callable[_P, _T],
    *,
    in_dims: PyTree[DimsSpec] = _SCALAR,
    out_dims: Union[Dims, Iterable[Dims]] = _SCALAR,
) -> Callable[_P, _T]:
    ...


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
    wrap_out = isinstance(out_dims, Optional[str])
    vectorize_wrapper = _VectorizeWrapper(
        _fun=fun,
        _in_dims=in_dims,
        _out_dims=(out_dims,) if wrap_out else tuple(out_dims),
        _signature=signature,
        _wrap_out=wrap_out,
    )

    return module_update_wrapper(vectorize_wrapper)
