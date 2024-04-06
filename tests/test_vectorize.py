from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import pytest
from equinox import Module
from equinox._vectorize import filter_vectorize
from equinox.nn import Linear
from jaxtyping import Array, PRNGKeyArray, Scalar

from .helpers import tree_allclose


@pytest.mark.parametrize("shape", [(), (2,), (3, 2)])
def test_scalar_kernel(shape):
    @eqx.filter_vectorize()
    def f(a: Scalar, b: Scalar) -> Scalar:
        return a + b

    # works with same batch shape
    # works when all passed as args
    a, b = jnp.full(shape, 1), jnp.full(shape, 2)
    out = f(a, b)
    assert tree_allclose(out, jnp.full(shape, 3))

    # works with broadcast-able shapes
    # works when one passed as kwarg
    a, b = jnp.full((5, *shape), 1), jnp.full(shape, 2)
    out = f(a, b=b)
    assert tree_allclose(out, jnp.full((5, *shape), 3))

    # works when all passed as kwargs
    a, b = jnp.full((5, *shape), 1), jnp.full((8, 5, *shape), 2)
    out = f(a=a, b=b)
    assert tree_allclose(out, jnp.full((8, 5, *shape), 3))

    # raises ValueError for unbroadcast-able shapes
    a, b = jnp.full((5, *shape), 1), jnp.full((*shape, 8), 2)
    with pytest.raises(ValueError):
        f(a, b)


@pytest.mark.parametrize("exclude", [True, False])
@pytest.mark.parametrize("mangle", [True, False])
@pytest.mark.parametrize("shape", [(), (2,), (3, 2)])
@pytest.mark.parametrize("use_bias", [True, False])
def test_module_args(
    exclude: bool, mangle: bool, shape: tuple[int, ...], use_bias: bool
):
    class M(eqx.Module):
        weights: Array = eqx.field(dims="(m,n)")
        bias: Optional[Array] = eqx.field(dims=eqx.dims_spec("(m)", exclude=exclude))
        use_bias: bool = eqx.field(static=True)

    @eqx.filter_vectorize(in_dims=(eqx.dims_spec(mangle=mangle), "(n)"), out_dims="(m)")
    def f(m: M, x: Array) -> Array:
        x = m.weights @ x
        if m.use_bias:
            x = x + m.bias
        return x

    m = M(
        weights=jnp.ones((*shape, 8, 5)),
        bias=jnp.ones((*shape, 8)) if use_bias else None,
        use_bias=use_bias,
    )

    # works on unbatched data
    # works when all passed as args
    # doesn't work when excluding bias array
    x = jnp.ones(5)
    if exclude and use_bias and shape != ():
        with pytest.raises(ValueError):
            f(m, x)
        return
    out = f(m, x)
    assert tree_allclose(out, jnp.full((*shape, 8), 5.0) + float(use_bias))

    # works on data batched same as module
    # works when one passed as kwarg
    x = jnp.ones((*shape, 5))
    out = f(m, x=x)
    assert tree_allclose(out, jnp.full((*shape, 8), 5.0) + float(use_bias))

    # works when data braodcast-able with module batching
    x = jnp.ones((13, *shape, 5))
    out = f(m, x)
    assert tree_allclose(out, jnp.full((13, *shape, 8), 5.0) + float(use_bias))

    # raises ValueError when both passed as kwargs
    # in_dims not tree mappable to dict
    with pytest.raises(ValueError):
        f(m=m, x=x)

    x = jnp.ones(8)
    if mangle:
        # when mangled incorrect dims cause TypeError at `m.weights @ x`
        with pytest.raises(TypeError):
            f(m, x)
    else:
        # when not mangled incorrect dims cause
        # ValueError in `jax.numpy.vectorize`
        with pytest.raises(ValueError):
            f(m, x)

    # raises ValueError with unbroadcast-able data
    x = jnp.ones(())
    with pytest.raises(ValueError):
        f(m, x)


class Result(Module):
    val: Array


@filter_vectorize(in_dims="(2)")
def make(key: PRNGKeyArray) -> Linear:
    return Linear(3, 5, key=key)


@filter_vectorize(in_dims=("preserve", "(n)"), out_dims="(m)")
def evaluate(model, x) -> Array:
    return model(x)


@filter_vectorize(in_dims=("mangle", "(n)"))
def evaluate_result(model, x) -> Result:
    return Result(model(x))


def main():
    key = jrandom.PRNGKey(0)
    ensemble_shape = (10,)
    data_batch_shape = (20, 10)

    key, subkey = jrandom.split(key)
    subkeys = jrandom.split(subkey, ensemble_shape)
    model = make(subkeys)
    print(model)

    key, subkey = jrandom.split(key)
    data = jrandom.normal(subkey, (*data_batch_shape, 3))

    eval_ = evaluate(model, data)
    result = evaluate_result(model, data)

    print(result)
    assert jnp.equal(eval_, result.val).all()


if __name__ == "__main__":
    main()
