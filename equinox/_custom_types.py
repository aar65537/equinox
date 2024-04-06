from typing import Any, Callable, Optional, Union

from jaxtyping import PyTree

from ._doc_utils import doc_repr


sentinel: Any = doc_repr(object(), "sentinel")
Dims = Optional[str]
DimsSpec = Union[Dims, Callable[[Any], PyTree[Dims]]]
