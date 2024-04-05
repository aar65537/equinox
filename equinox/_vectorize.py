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
from ._filters import combine, is_array, partition
from ._module import Module, module_update_wrapper, Partial


_P = ParamSpec("_P")
_T = TypeVar("_T")


class _CoreDimsOptions(StrEnum):
    EXCLUDE = auto()
    MANGLE = auto()
    PRESERVE = auto()
    SCALAR = "()"


CoreDims = str
CoreDimsSpec = Union[CoreDims, Callable[[Any], PyTree[CoreDims]]]

# See http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
_DIMENSION_NAME = r"\w+"
_CORE_DIMENSION_LIST = "(?:{0:}(?:,{0:})*)?".format(_DIMENSION_NAME)
_ARGUMENT = rf"\({_CORE_DIMENSION_LIST}\)"


def _is_none(x: Any) -> bool:
    return x is None


def _validate_core_dims(core_dims: CoreDims) -> None:
    if not re.match(_ARGUMENT, core_dims):
        raise ValueError(f"not a valid gufunc core dimension: {core_dims}")


def _parse_core_dims(core_dims: CoreDims) -> list[str]:
    _validate_core_dims(core_dims)
    return re.findall(_DIMENSION_NAME, core_dims)


@dataclasses.dataclass
class _mangle:
    _id: str

    def __call__(self, core_dims: str) -> str:
        _core_dims = ",".join(self._id + "_" + cd for cd in _parse_core_dims(core_dims))
        return "(" + _core_dims + ")"


@dataclasses.dataclass(frozen=True)
class core_dims_spec:
    array_core_dims: CoreDims = _CoreDimsOptions.SCALAR
    dataclass_core_dims: CoreDims = _CoreDimsOptions.MANGLE
    else_core_dims: CoreDims = _CoreDimsOptions.EXCLUDE

    def __call__(self, x: Any) -> PyTree[CoreDims]:
        if isinstance(x, Array):
            core_dims = self.array_core_dims
        elif dataclasses.is_dataclass(x):
            if self.dataclass_core_dims not in {
                _CoreDimsOptions.MANGLE,
                _CoreDimsOptions.PRESERVE,
            }:
                raise ValueError(
                    "`in_core_dims` for dataclasses must be 'mangle' or 'preserve'."
                )

            core_dims = []
            for field in dataclasses.fields(x):
                if field.metadata.get("static", False):
                    continue
                try:
                    value = x.__dict__[field.name]
                except KeyError:
                    continue
                core_dims_spec = field.metadata.get("core_dims", self.__class__())
                core_dims.append(_resolve_core_dims(value, core_dims_spec))

            if self.dataclass_core_dims == _CoreDimsOptions.MANGLE:
                core_dims = jtu.tree_map(_mangle(str(id(x))), core_dims)
            print(core_dims)
        else:
            core_dims = self.else_core_dims

        return core_dims


def _resolve_core_dims_spec(
    _core_dims_spec: CoreDimsSpec, elem: Any
) -> PyTree[CoreDims]:
    if isinstance(_core_dims_spec, str):
        if _core_dims_spec == _CoreDimsOptions.EXCLUDE:
            _core_dims_spec = core_dims_spec(
                array_core_dims=_core_dims_spec,
                dataclass_core_dims=_core_dims_spec,
                else_core_dims=_core_dims_spec,
            )
        elif (
            _core_dims_spec == _CoreDimsOptions.MANGLE
            or _core_dims_spec == _CoreDimsOptions.PRESERVE
        ):
            _core_dims_spec = core_dims_spec(
                array_core_dims=_CoreDimsOptions.SCALAR,
                dataclass_core_dims=_core_dims_spec,
                else_core_dims=_CoreDimsOptions.EXCLUDE,
            )
        else:
            _validate_core_dims(_core_dims_spec)
            _core_dims_spec = core_dims_spec(
                array_core_dims=_core_dims_spec,
                dataclass_core_dims=_CoreDimsOptions.MANGLE,
                else_core_dims=_CoreDimsOptions.EXCLUDE,
            )

    if not callable(_core_dims_spec):
        raise ValueError(
            "`in_core_dims` must be a PyTree of strings and callables only."
        )

    return jtu.tree_map(_core_dims_spec, elem, is_leaf=dataclasses.is_dataclass)


def _resolve_core_dims(
    pytree: PyTree[Any], core_dims_spec: PyTree[CoreDimsSpec]
) -> PyTree[CoreDims]:
    return jtu.tree_map(_resolve_core_dims_spec, core_dims_spec, pytree)


def _signature(
    in_core_dims: PyTree[CoreDims],
    out_core_dims: PyTree[CoreDims],
) -> str:
    in_core_dims_leaves, _ = jtu.tree_flatten(in_core_dims)
    out_core_dims_leaves, _ = jtu.tree_flatten(out_core_dims)

    in_signature = ",".join(in_core_dims_leaves)
    out_signature = ",".join(out_core_dims_leaves)
    signature = "->".join((in_signature, out_signature))

    return signature


class _VectorizeWrapper(Module):
    _fun: Callable
    _in_core_dims: PyTree[CoreDimsSpec]
    _out_core_dims: PyTree[CoreDims]

    @property
    def __wrapped__(self):
        return self._fun

    def __call__(self, /, *args, **kwargs):
        if args and kwargs:
            in_ = (*args, kwargs)
        elif args:
            in_ = args
        elif kwargs:
            in_ = kwargs
        else:
            raise ValueError(
                "Vectorized function must be called with `args` or `kwargs`"
            )

        dynamic_in, static_in = partition(in_, is_array)
        dynamic_in_leaves, in_treedef = jtu.tree_flatten(dynamic_in, is_leaf=_is_none)

        def _fun_wrapper(*_dynamic_in_leaves):
            _dynamic_in = jtu.tree_unflatten(in_treedef, _dynamic_in_leaves)
            _in_ = combine(_dynamic_in, static_in)

            if args and kwargs:
                *_args, _kwargs = _in_
            elif args:
                _args, _kwargs = _in_, {}
            elif kwargs:
                _args, _kwargs = (), _in_
            else:
                assert False, "unreachable"

            return self._fun(*_args, **_kwargs)

        in_core_dims = _resolve_core_dims(in_, self._in_core_dims)
        signature = _signature(in_core_dims, self._out_core_dims)
        return jnp.vectorize(_fun_wrapper, signature=signature)(*dynamic_in_leaves)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return Partial(self, instance)


@overload
def filter_vectorize(
    *,
    in_core_dims: PyTree[CoreDimsSpec] = core_dims_spec(),
    out_core_dims: PyTree[CoreDims] = _CoreDimsOptions.SCALAR,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    ...


@overload
def filter_vectorize(
    fun: Callable[_P, _T],
    *,
    in_core_dims: PyTree[CoreDimsSpec] = core_dims_spec(),
    out_core_dims: PyTree[CoreDims] = _CoreDimsOptions.SCALAR,
) -> Callable[_P, _T]:
    ...


def filter_vectorize(
    fun=sentinel,
    in_core_dims=core_dims_spec(),
    out_core_dims=_CoreDimsOptions.SCALAR,
):
    if fun is sentinel:
        return ft.partial(
            filter_vectorize,
            in_core_dims=in_core_dims,
            out_core_dims=out_core_dims,
        )

    vectorize_wrapper = _VectorizeWrapper(
        _fun=fun,
        _in_core_dims=in_core_dims,
        _out_core_dims=out_core_dims,
    )
    return module_update_wrapper(vectorize_wrapper)
