import jax.numpy as jnp
import jax.random as jrandom
from equinox import Module
from equinox._vectorize import filter_vectorize
from equinox.nn import Linear
from jaxtyping import Array, PRNGKeyArray


class Result(Module):
    val: Array


@filter_vectorize(in_core_dims="(2)")
def make(key: PRNGKeyArray) -> Linear:
    return Linear(3, 5, key=key)


@filter_vectorize(in_core_dims=("preserve", "(n)"), out_core_dims="(m)")
def evaluate(model, x) -> Array:
    return model(x)


@filter_vectorize(in_core_dims=("mangle", "(n)"))
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
