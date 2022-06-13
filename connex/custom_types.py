# Modified from https://github.com/patrick-kidger/equinox/blob/main/equinox/custom_types.py

import inspect
import typing
from typing import Generic, Tuple, TypeVar, Union


# Custom flag we set when generating documentation.
if getattr(typing, "GENERATING_DOCUMENTATION", False):

    def _item_to_str(item: Union[str, type, slice]) -> str:
        if isinstance(item, slice):
            if item.step is not None:
                raise NotImplementedError
            return _item_to_str(item.start) + ": " + _item_to_str(item.stop)
        elif item is ...:
            return "..."
        elif inspect.isclass(item):
            return item.__name__
        else:
            return repr(item)

    def _maybe_tuple_to_str(
        item: Union[str, type, slice, Tuple[Union[str, type, slice], ...]]
    ) -> str:
        if isinstance(item, tuple):
            if len(item) == 0:
                # Explicit brackets
                return "()"
            else:
                # No brackets
                return ", ".join([_item_to_str(i) for i in item])
        else:
            return _item_to_str(item)

    _Annotation = TypeVar("_Annotation")

    class _Array(Generic[_Annotation]):
        pass

    _Array.__module__ = "builtins"
    _Array.__qualname__ = "Array"

    class Array:
        def __class_getitem__(cls, item):
            class X:
                pass

            X.__module__ = "builtins"
            X.__qualname__ = _maybe_tuple_to_str(item)
            return _Array[X]

    Array.__module__ = "builtins"

else:

    class Array:
        def __class_getitem__(cls, item):
            return Array