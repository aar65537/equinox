import functools as ft
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import pytest
from equinox.internal import GetKey
from jaxtyping import Array, PRNGKeyArray, Scalar

from .helpers import tree_allclose


@pytest.mark.parametrize("shape", [(), (3, 2)])
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


@pytest.mark.parametrize("shape", [(), (3, 2)])
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
@pytest.mark.parametrize("shape", [(), (3, 2)])
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


@pytest.mark.parametrize("shape", [(), (3, 2)])
@pytest.mark.parametrize("use_bias", [True, False])
def test_module_output(getkey: GetKey, shape: tuple[int, ...], use_bias: bool):
    class M(eqx.Module):
        weights: Array
        bias: Optional[Array]
        use_bias: bool = eqx.field(static=True)

    @eqx.filter_vectorize(in_dims=("2"))
    def f(key: PRNGKeyArray) -> M:
        weights = jrandom.normal(key, (8, 5))
        bias = jrandom.normal(key, (5,)) if use_bias else None
        return M(weights=weights, bias=bias, use_bias=use_bias)

    keys = jrandom.split(getkey(), shape)
    m = f(keys)

    assert m.weights.shape == (*shape, 8, 5)
    if use_bias:
        assert m.use_bias
        assert m.bias is not None
        assert m.bias.shape == (*shape, 5)
    else:
        assert not m.use_bias
        assert m.bias is None


@pytest.mark.parametrize("shape", [(), (3, 2)])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("container", [tuple, list])
def test_multiple_output(
    getkey: GetKey, shape: tuple[int, ...], use_bias: bool, container
):
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

    keys = jrandom.split(getkey(), shape)
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
@pytest.mark.parametrize("shape", [(), (3, 2)])
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
