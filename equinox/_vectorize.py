import dataclasses
import functools as ft
import re
from collections.abc import Callable
from enum import auto, StrEnum
from typing import Any, overload, ParamSpec, TypeVar, Union

import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PyTree

from ._custom_types import sentinel
from ._filters import combine, filter, is_array, partition
from ._module import Module, module_update_wrapper, Partial


_P = ParamSpec("_P")
_T = TypeVar("_T")


class _DimsOptions(StrEnum):
    EXCLUDE = auto()
    MANGLE = auto()
    PRESERVE = auto()
    SCALAR = "()"


Dims = str
DimsSpec = Union[Dims, Callable[[Any], PyTree[Dims]]]

_DIMENSION_NAME = r"\w+"
_DIMENSION_LIST = "(?:{0:}(?:,{0:})*)?".format(_DIMENSION_NAME)
_ARGUMENT = rf"\({_DIMENSION_LIST}\)"


def _parse_dims(dims: Dims) -> tuple[str, ...]:
    if not re.match(_ARGUMENT, dims) and dims != _DimsOptions.EXCLUDE:
        raise ValueError(f"{dims} is not a valid gufunc dimension signature.")

    return tuple(re.findall(_DIMENSION_NAME, dims))


@dataclasses.dataclass
class _mangle_dims:
    _id: str

    def __call__(self, dims: Dims) -> Dims:
        return "(" + ",".join(self._id + "_" + dim for dim in _parse_dims(dims)) + ")"


@dataclasses.dataclass
class _batch_dims:
    _batch: Dims

    def __call__(self, dims: Dims) -> Dims:
        return "(" + ",".join((*_parse_dims(self._batch), *_parse_dims(dims))) + ")"


@dataclasses.dataclass(frozen=True)
class dims_spec:
    array_core_dims: Dims = _DimsOptions.SCALAR
    dataclass_core_dims: Dims = _DimsOptions.MANGLE
    else_core_dims: Dims = _DimsOptions.EXCLUDE

    def __call__(self, x: Any) -> PyTree[Dims]:
        if isinstance(x, Array):
            dims = self.array_core_dims
        elif dataclasses.is_dataclass(x):
            mode, *batch_dims = self.dataclass_core_dims.split("_")
            batch_dims = batch_dims[0] if batch_dims else "()"
            if mode not in {_DimsOptions.MANGLE, _DimsOptions.PRESERVE}:
                raise ValueError(
                    "`in_dims` for dataclasses must be 'mangle' or 'preserve'."
                )

            dims = []
            for field in dataclasses.fields(x):
                if field.metadata.get("static", False):
                    continue
                try:
                    value = x.__dict__[field.name]
                except KeyError:
                    continue
                dims_spec = field.metadata.get("dims", self.__class__())
                resolved_core_dims = _resolve_dims(value, dims_spec)
                dims.append(resolved_core_dims)
            if mode == _DimsOptions.MANGLE:
                dims = jtu.tree_map(_mangle_dims(str(id(x))), dims)
            dims = jtu.tree_map(_batch_dims(batch_dims), dims)
        else:
            dims = self.else_core_dims

        return dims


def _is_none(x: Any) -> bool:
    return x is None


def _is_dataclass_or_none(x: Any) -> bool:
    return dataclasses.is_dataclass(x) or _is_none(x)


def _resolve_dims_spec(_dims_spec: DimsSpec, elem: Any) -> PyTree[Dims]:
    if isinstance(_dims_spec, Dims):
        if _dims_spec == _DimsOptions.EXCLUDE:
            _dims_spec = dims_spec(
                array_core_dims=_DimsOptions.EXCLUDE,
                dataclass_core_dims=_DimsOptions.EXCLUDE,
                else_core_dims=_DimsOptions.EXCLUDE,
            )
        elif _dims_spec.split("_")[0] in {
            _DimsOptions.MANGLE,
            _DimsOptions.PRESERVE,
        }:
            _dims_spec = dims_spec(
                array_core_dims=_DimsOptions.SCALAR,
                dataclass_core_dims=_dims_spec,
                else_core_dims=_DimsOptions.EXCLUDE,
            )
        else:
            _dims_spec = dims_spec(
                array_core_dims=_dims_spec,
                dataclass_core_dims=_DimsOptions.MANGLE + "_" + _dims_spec,
                else_core_dims=_DimsOptions.EXCLUDE,
            )

    if not callable(_dims_spec):
        raise ValueError("`in_dims` must be a PyTree of strings and callables only.")

    return jtu.tree_map(_dims_spec, elem, is_leaf=_is_dataclass_or_none)


def _resolve_dims(pytree: PyTree[Any], dims_spec: PyTree[DimsSpec]) -> PyTree[Dims]:
    return jtu.tree_map(_resolve_dims_spec, dims_spec, pytree)


def _is_exclude(x: Any, dims: Dims) -> bool:
    return not is_array(x) or dims == _DimsOptions.EXCLUDE


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
    in_dims_leaves = filter(in_dims_leaves, filter_leaves, True, _DimsOptions.SCALAR)
    in_dims = ",".join(in_dims_leaves)
    signature = "->".join((in_dims, out_dims))
    return signature


class _VectorizeWrapper(Module):
    _fun: Callable
    _in_dims: PyTree[DimsSpec]
    _out_dims: PyTree[Dims]

    @property
    def __wrapped__(self):
        return self._fun

    def __call__(self, /, *args, **kwargs):
        in_ = _combine_args_kwargs(args, kwargs)
        in_dims = _resolve_dims(in_, self._in_dims)
        in_leaves, in_treedef = jtu.tree_flatten(in_, is_leaf=_is_none)
        in_dims_leaves, _ = jtu.tree_flatten(in_dims)
        filter_leaves = jtu.tree_map(
            _is_exclude, in_leaves, in_dims_leaves, is_leaf=_is_none
        )
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
    *, in_dims: PyTree[DimsSpec] = dims_spec(), out_dims: str = _DimsOptions.SCALAR
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    ...


@overload
def filter_vectorize(
    fun: Callable[_P, _T],
    *,
    in_dims: PyTree[DimsSpec] = dims_spec(),
    out_dims: str = _DimsOptions.SCALAR,
) -> Callable[_P, _T]:
    ...


def filter_vectorize(fun=sentinel, in_dims=dims_spec(), out_dims=_DimsOptions.SCALAR):
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
