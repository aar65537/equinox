import functools as ft
from typing import Literal, Optional, Union

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import pytest
from jaxtyping import Array, PRNGKeyArray, Scalar

from .helpers import tree_allclose


@pytest.mark.parametrize("shape", [(), (2,), (3, 2)])
def test_scalar_fn(shape: tuple[int, ...]):
    @eqx.filter_vectorize()
    def f(a: Scalar, b: Scalar) -> Scalar:
        return a + b

    # works with same loop dims
    # works with all input passed as args
    a, b = jnp.full(shape, 1), jnp.full(shape, 2)
    out = f(a, b)
    assert tree_allclose(out, jnp.full(shape, 3))

    # works with broadcastable loop dims
    # works with one input passed as kwarg
    a, b = jnp.full((5, *shape), 1), jnp.full(shape, 2)
    out = f(a, b=b)
    assert tree_allclose(out, jnp.full((5, *shape), 3))

    # works with all input passed as kwargs
    a, b = jnp.full((5, *shape), 1), jnp.full((8, 5, *shape), 2)
    out = f(a=a, b=b)
    assert tree_allclose(out, jnp.full((8, 5, *shape), 3))

    # raises ValueError with unbroadcastable loop dims
    a, b = jnp.full((5, *shape), 1), jnp.full((*shape, 8), 2)
    with pytest.raises(ValueError):
        f(a, b)


@pytest.mark.parametrize("shape", [(), (2,), (3, 2)])
def test_array_fn(shape: tuple[int, ...]):
    @eqx.filter_vectorize(in_dims=("m n", "n"), out_dims="m")
    def f(a: Array, b: Array) -> Array:
        return a @ b

    # works with same loop dims
    # works with all input passed as args
    a, b = jnp.ones((*shape, 8, 5)), jnp.ones((*shape, 5))
    out = f(a, b)
    assert tree_allclose(out, jnp.full((*shape, 8), 5.0))

    # works with boradcastable loop dims
    # works with one input passed as kwarg
    a, b = jnp.ones((*shape, 8, 5)), jnp.ones((5))
    out = f(a, b=b)
    assert tree_allclose(out, jnp.full((*shape, 8), 5.0))

    # works with all input passed as kwargs
    a, b = jnp.ones((13, *shape, 8, 5)), jnp.ones((21, 13, *shape, 5))
    out = f(a=a, b=b)
    assert tree_allclose(out, jnp.full((21, 13, *shape, 8), 5.0))

    # raises ValueError with inconsistent core dims
    a, b = jnp.ones((*shape, 8, 5)), jnp.ones((*shape, 8))
    with pytest.raises(ValueError):
        f(a, b)

    # raises ValueError with unboradcastable loop dims
    a, b = jnp.ones((21, *shape, 8, 5)), jnp.ones((*shape, 13, 5))
    with pytest.raises(ValueError):
        f(a, b)


@pytest.mark.parametrize("exclude", [True, False])
@pytest.mark.parametrize("mangle", [True, False])
@pytest.mark.parametrize("shape", [(), (2,), (3, 2)])
@pytest.mark.parametrize("use_bias", [True, False])
def test_module_input(
    exclude: bool, mangle: bool, shape: tuple[int, ...], use_bias: bool
):
    class M(eqx.Module):
        weights: Array = eqx.field(dims="m n")
        bias: Optional[Array] = eqx.field(dims=None if exclude else "m")
        use_bias: bool = eqx.field(static=True)

    @eqx.filter_vectorize(in_dims=(eqx.dims_spec(mangle=mangle), "n"), out_dims="m")
    def f(m: M, x: Array) -> Array:
        x = m.weights @ x
        if m.use_bias:
            x = x + m.bias
        return x

    # M with unbatched bias
    m = M(
        weights=jnp.ones((*shape, 8, 5)),
        bias=jnp.ones(8) if use_bias else None,
        use_bias=use_bias,
    )

    # works with core dims
    # works with all input passed as args
    x = jnp.ones(5)
    out = f(m, x)
    assert tree_allclose(out, jnp.full((*shape, 8), 5.0) + float(use_bias))

    # works with with same loop dims
    # works with one input passed as kwarg
    x = jnp.ones((*shape, 5))
    out = f(m, x=x)
    assert tree_allclose(out, jnp.full((*shape, 8), 5.0) + float(use_bias))

    # works with broadcastable loop dims
    # works with all input passed as kwargs
    x = jnp.ones((13, *shape, 5))
    out = f(m=m, x=x)
    assert tree_allclose(out, jnp.full((13, *shape, 8), 5.0) + float(use_bias))

    x = jnp.ones(8)
    if mangle:
        # raises TypeError when mangled with inconsistent core dims
        with pytest.raises(TypeError):
            f(m, x)
    else:
        # raises ValueError when not mangled with inconsistent core dims
        with pytest.raises(ValueError):
            f(m, x)

    # raises ValueError with unbroadcastable loop dims
    x = jnp.ones(())
    with pytest.raises(ValueError):
        f(m, x)

    # M with batched bias
    m = M(
        weights=jnp.ones((*shape, 8, 5)),
        bias=jnp.ones((*shape, 8)) if use_bias else None,
        use_bias=use_bias,
    )

    # raises ValueError with excluded batched bias
    x = jnp.ones(5)
    if shape == () or not exclude or not use_bias:
        f(m, x)
    else:
        with pytest.raises(ValueError):
            f(m, x)


@pytest.mark.parametrize("shape", [(), (2,), (3, 2)])
@pytest.mark.parametrize("use_bias", [True, False])
def test_module_output(shape: tuple[int, ...], use_bias: bool):
    class M(eqx.Module):
        weights: Array
        bias: Optional[Array]
        use_bias: bool = eqx.field(static=True)

    @eqx.filter_vectorize(in_dims=("2"))
    def f(key: PRNGKeyArray) -> M:
        weights = jrandom.normal(key, (8, 5))
        bias = jrandom.normal(key, (5,)) if use_bias else None
        return M(weights=weights, bias=bias, use_bias=use_bias)

    keys = jrandom.split(jrandom.PRNGKey(0), shape)
    m = f(keys)

    assert m.weights.shape == (*shape, 8, 5)
    if use_bias:
        assert m.use_bias
        assert m.bias is not None
        assert m.bias.shape == (*shape, 5)
    else:
        assert not m.use_bias
        assert m.bias is None


@pytest.mark.parametrize("shape", [(), (2,), (3, 2)])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("container", [tuple, list])
def test_multiple_output(shape: tuple[int, ...], use_bias: bool, container):
    class M(eqx.Module):
        weights: Array
        bias: Optional[Array]
        use_bias: bool = eqx.field(static=True)

    @eqx.filter_vectorize(in_dims=("2"), out_dims=("", "2", None))
    def f(key: PRNGKeyArray) -> tuple[M, PRNGKeyArray, Array]:
        weights = jrandom.normal(key, (8, 5))
        bias = jrandom.normal(key, (5,)) if use_bias else None
        return container(
            [M(weights=weights, bias=bias, use_bias=use_bias), key, jnp.asarray(0)]
        )

    keys = jrandom.split(jrandom.PRNGKey(0), shape)
    out = f(keys)
    m, _keys, zero = out

    assert isinstance(out, container)
    assert m.weights.shape == (*shape, 8, 5)
    assert tree_allclose(_keys, keys)
    assert tree_allclose(zero, jnp.asarray(0))
    if use_bias:
        assert m.use_bias
        assert m.bias is not None
        assert m.bias.shape == (*shape, 5)
    else:
        assert not m.use_bias
        assert m.bias is None


@pytest.mark.parametrize("call", [False, True])
@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("shape", [(), (2,), (3, 2)])
def test_methods(call: bool, outer: bool, shape: tuple[int, ...]):
    vectorize = ft.partial(
        eqx.filter_vectorize,
        in_dims=(eqx.dims_spec(mangle=False), "n"),
        out_dims="m",
    )

    class M(eqx.Module):
        weights: Array = eqx.field(dims="m n")
        bias: Optional[Array] = eqx.field(dims="m")

        if call:

            def __call__(self, x: Array) -> Array:
                return self.weights @ x + self.bias

            if not outer:
                __call__ = vectorize(__call__)
        else:

            def method(self, x: Array) -> Array:
                return self.weights @ x + self.bias

            if not outer:
                method = vectorize(method)

    m = M(jnp.ones((8, 5)), jnp.ones(8))
    x = jnp.ones((*shape, 5))
    y = jnp.full((*shape, 8), 6.0)

    if call:
        if outer:
            tree_allclose(vectorize(M.__call__)(m, x), y)
        else:
            tree_allclose(m(x), y)
    else:
        if outer:
            tree_allclose(vectorize(M.method)(m, x), y)
        else:
            tree_allclose(m.method(x), y)


@pytest.mark.parametrize("data_shape", [(), (2,), (3, 2)])
@pytest.mark.parametrize("ensemble_shape", [(), (2,), (3, 2)])
@pytest.mark.parametrize("in_features", [5, "scalar"])
@pytest.mark.parametrize("out_features", [8, "scalar"])
@pytest.mark.parametrize("use_bias", [True, False])
def test_linear_ensemble(
    data_shape: tuple[int, ...],
    ensemble_shape: tuple[int, ...],
    in_features: Union[int, Literal["scalar"]],
    out_features: Union[int, Literal["scalar"]],
    use_bias: bool,
):
    @eqx.filter_vectorize(in_dims="2")
    def make(key: PRNGKeyArray) -> eqx.nn.Linear:
        return eqx.nn.Linear(in_features, out_features, use_bias, key=key)

    @eqx.filter_vectorize(
        in_dims=(
            eqx.dims_spec(mangle=False),
            "" if in_features == "scalar" else "n",
        ),
        out_dims="" if out_features == "scalar" else "m",
    )
    def evaluate(model, x) -> Array:
        return model(x)

    keys = jrandom.split(jrandom.PRNGKey(0), ensemble_shape)
    model = make(keys)
    assert model.weight.shape == (
        *ensemble_shape,
        1 if out_features == "scalar" else out_features,
        1 if in_features == "scalar" else in_features,
    )
    if use_bias:
        assert model.bias is not None
        assert model.bias.shape == (
            *ensemble_shape,
            1 if out_features == "scalar" else out_features,
        )

    if in_features == "scalar":
        x = jnp.ones(data_shape)
    else:
        x = jnp.ones((*data_shape, in_features))
    out = evaluate(model, x)

    loop_dims = max(data_shape, ensemble_shape, key=len)
    if out_features == "scalar":
        assert out.shape == loop_dims
    else:
        assert out.shape == (*loop_dims, out_features)


@pytest.mark.parametrize("data_shape", [(), (2,), (3, 2)])
@pytest.mark.parametrize("ensemble_shape", [(), (2,), (3, 2)])
@pytest.mark.parametrize("in_size", [5, "scalar"])
@pytest.mark.parametrize("out_size", [8, "scalar"])
def test_mlp_ensemble(
    data_shape: tuple[int, ...],
    ensemble_shape: tuple[int, ...],
    in_size: Union[int, Literal["scalar"]],
    out_size: Union[int, Literal["scalar"]],
):
    @eqx.filter_vectorize(in_dims="2")
    def make(key: PRNGKeyArray) -> eqx.nn.MLP:
        return eqx.nn.MLP(in_size, out_size, 13, 4, key=key)

    @eqx.filter_vectorize(
        in_dims=("", "" if in_size == "scalar" else "n"),
        out_dims="" if out_size == "scalar" else "m",
    )
    def evaluate(model, x) -> Array:
        return model(x)

    keys = jrandom.split(jrandom.PRNGKey(0), ensemble_shape)
    model = make(keys)

    for layer in model.layers:
        assert layer.weight.shape[:-2] == ensemble_shape
        assert layer.bias is not None
        assert layer.bias.shape[:-1] == ensemble_shape

    if in_size == "scalar":
        x = jnp.ones(data_shape)
    else:
        x = jnp.ones((*data_shape, in_size))

    out = evaluate(model, x)
    loop_dims = max(data_shape, ensemble_shape, key=len)

    if out_size == "scalar":
        assert out.shape == loop_dims
    else:
        assert out.shape == (*loop_dims, out_size)
