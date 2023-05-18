__all__ = ["str_mapping"]

from collections.abc import Mapping

from coola.utils.format import str_indent


def str_mapping(mapping: Mapping, sorted_keys: bool = False, num_spaces: int = 2) -> str:
    r"""Computes a torch-like (``torch.nn.Module``) string representation of a
    mapping.

    Args:
    ----
        mapping (``Mapping``): Specifies the mapping.
        sorted_keys (bool, optional): Specifies if the key of the dict
            are sorted or not. Default: ``False``
        num_spaces (int, optional): Specifies the number of spaces
            used for the indentation. Default: ``2``.

    Returns:
    -------
        str: The string representation of the mapping.

    Example usage:

    .. code-block:: python

        >>> from redcat.utils.format import str_mapping
        >>> str_mapping({'key1': 'abc', 'key2': 'something\nelse'})
        (key1) abc
        (key2) something
          else
    """
    lines = []
    for key, value in sorted(mapping.items()) if sorted_keys else mapping.items():
        lines.append(f"({key}) {str_indent(value, num_spaces=num_spaces)}")
    return "\n".join(lines)
