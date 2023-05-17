from redcat.format import str_mapping

#################################
#     Tests for str_mapping     #
#################################


def test_str_mapping_empty() -> None:
    assert str_mapping({}) == ""


def test_str_mapping_1_item() -> None:
    assert str_mapping({"key": "value"}) == "(key) value"


def test_str_mapping_2_items() -> None:
    assert str_mapping({"key1": "value1", "key2": "value2"}) == "(key1) value1\n(key2) value2"


def test_str_mapping_sorted_values_true() -> None:
    assert (
        str_mapping({"key2": "value2", "key1": "value1"}, sorted_keys=True)
        == "(key1) value1\n(key2) value2"
    )


def test_str_mapping_sorted_values_false() -> None:
    assert str_mapping({"key2": "value2", "key1": "value1"}) == "(key2) value2\n(key1) value1"


def test_str_mapping_2_items_multiple_line() -> None:
    assert (
        str_mapping({"key1": "abc", "key2": "something\nelse"})
        == "(key1) abc\n(key2) something\n  else"
    )
