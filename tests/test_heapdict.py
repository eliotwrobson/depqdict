# Tests based on https://github.com/nanouasyn/heapdict/blob/main/test/test_heapdict.py
import copy
import operator
from collections import OrderedDict
from contextlib import contextmanager

import hypothesis
import pytest
from hypothesis import given
from hypothesis import strategies as st

from depqdict import DepqDict
from tests.utils import assert_heapdict_is_empty, check_heapdict_invariants


@contextmanager
def heapdict_not_changes(heapdict: DepqDict):
    heapdict_copy = heapdict.copy()
    try:
        yield heapdict
    except BaseException:
        raise
    assert heapdict._mapping == heapdict_copy._mapping
    assert heapdict._heap == heapdict_copy._heap


def assert_args_are_equivalent_by_function(func, a, b) -> None:
    a_result, a_error = None, None
    try:
        a_result = func(a)
    except Exception as e:
        a_error = e

    b_result, b_error = None, None
    try:
        b_result = func(b)
    except Exception as e:
        b_error = e

    assert a_result == b_result
    assert type(a_error) is type(b_error)


def test_create_empty() -> None:
    assert_heapdict_is_empty(DepqDict())


def test_create_from_incompatible() -> None:
    with pytest.raises(TypeError):
        _ = DepqDict(42)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        _ = DepqDict([1, 2, 3])  # type: ignore[arg-type, list-item]


@given(pairs=st.lists(st.tuples(st.integers(), st.integers())))
def test_create_from_pairs(pairs: list[tuple[int, int]]) -> None:
    pairs = []
    seen = set()
    for a, b in pairs:
        if a not in seen:
            seen.add(a)
            pairs.append((a, b))

    heapdict: DepqDict[int, int] = DepqDict(pairs)
    check_heapdict_invariants(heapdict)

    expected = dict(pairs)
    assert heapdict == expected


@given(dictionary=st.dictionaries(st.integers(), st.integers()))
def test_create_from_dict(dictionary: dict[int, int]) -> None:
    heapdict = DepqDict(dictionary)
    check_heapdict_invariants(heapdict)

    assert heapdict == dictionary


@given(dictionary=st.dictionaries(st.integers(), st.integers()))
def test_create_from_heapdict(dictionary: dict[int, int]) -> None:
    another_heapdict: DepqDict = DepqDict(dictionary)
    check_heapdict_invariants(another_heapdict)

    heapdict: DepqDict = DepqDict(another_heapdict)
    check_heapdict_invariants(heapdict)

    assert dict(heapdict) == dict(another_heapdict)


@given(dictionary=st.dictionaries(st.integers(), st.integers()))
def test_create_from_insertions(dictionary: dict[int, int]) -> None:
    created = DepqDict(dictionary)

    filled: DepqDict[int, int] = DepqDict()
    for key, value in dictionary.items():
        filled[key] = value
    check_heapdict_invariants(filled)

    assert dict(created) == dict(filled)

    created_order = [created.pop_min_item()[1] for _ in range(len(created))]
    assert_heapdict_is_empty(created)

    filled_order = [filled.pop_min_item()[1] for _ in range(len(filled))]
    assert_heapdict_is_empty(filled)

    assert created_order == filled_order


@pytest.mark.parametrize("copy_func", [copy.copy, lambda x: x.copy()])
def test_copy(copy_func) -> None:
    original = DepqDict(zip("abcdef", [3, 3, 1, 6, 3, 4], strict=False))

    clone = copy_func(original)
    check_heapdict_invariants(original)

    assert original is not clone
    assert original == clone
    assert dict(original) == dict(clone)

    assert original._mapping == clone._mapping
    assert original._mapping is not clone._mapping
    assert original._heap == clone._heap
    assert original._heap is not clone._heap

    clone["x"] = 5
    check_heapdict_invariants(original)
    check_heapdict_invariants(clone)

    assert "x" in clone
    assert "x" not in original

    clone.pop("d")
    check_heapdict_invariants(original)
    check_heapdict_invariants(clone)

    assert "d" not in clone
    assert "d" in original

    clone["a"] = 100
    check_heapdict_invariants(original)
    check_heapdict_invariants(clone)

    assert clone["a"] == 100
    assert original["a"] != 100


@given(dictionary=st.dictionaries(st.integers(), st.integers(), min_size=1))
def test_pop_min(dictionary: dict[int, int]) -> None:
    heapdict = DepqDict(dictionary)

    with heapdict_not_changes(heapdict):
        key, priority = item = heapdict.min_item()

        assert key in heapdict
        assert heapdict[key] == priority

    extracted_item = heapdict.pop_min_item()
    check_heapdict_invariants(heapdict)

    with heapdict_not_changes(heapdict):
        assert key not in heapdict
        with pytest.raises(KeyError):
            _ = heapdict[key]

    assert extracted_item == item


@given(dictionary=st.dictionaries(st.integers(), st.integers(), min_size=1))
def test_pop_max(dictionary: dict[int, int]) -> None:
    heapdict = DepqDict(dictionary)

    with heapdict_not_changes(heapdict):
        key, priority = item = heapdict.max_item()

        assert key in heapdict
        assert heapdict[key] == priority

    extracted_item = heapdict.pop_max_item()
    check_heapdict_invariants(heapdict)

    with heapdict_not_changes(heapdict):
        assert key not in heapdict
        with pytest.raises(KeyError):
            _ = heapdict[key]

    assert extracted_item == item


@given(
    dictionary=st.dictionaries(st.integers(), st.integers(), min_size=1),
    random=st.randoms(),
)
def test_pop(dictionary: dict[int, int], random) -> None:
    heapdict = DepqDict(dictionary)

    key = random.choice(list(dictionary.items()))[0]
    hypothesis.note(f"{key = }")

    with heapdict_not_changes(heapdict):
        assert key in heapdict
        priority = heapdict[key]

    extracted_priority = heapdict.pop(key)
    check_heapdict_invariants(heapdict)

    with heapdict_not_changes(heapdict):
        assert key not in heapdict
        with pytest.raises(KeyError):
            _ = heapdict[key]

    assert extracted_priority == priority


@given(dictionary=st.dictionaries(st.integers(), st.integers(), min_size=1))
def test_pop_by_last_heap_index(dictionary: dict[int, int]) -> None:
    heapdict = DepqDict(dictionary)

    with heapdict_not_changes(heapdict):
        last_key = heapdict._heap[-1].key
        hypothesis.note(f"{last_key = }")

        assert last_key in heapdict
        priority = heapdict[last_key]

    extracted_priority = heapdict.pop(last_key)
    check_heapdict_invariants(heapdict)

    with heapdict_not_changes(heapdict):
        assert last_key not in heapdict
        with pytest.raises(KeyError):
            _ = heapdict[last_key]

    assert priority == extracted_priority


@given(dictionary=st.dictionaries(st.integers(), st.integers(), min_size=1))
def test_popitem(dictionary: dict[int, int]) -> None:
    heapdict = DepqDict(dictionary)

    with heapdict_not_changes(heapdict):
        key = next(reversed(heapdict._mapping))

        assert key in heapdict
        priority = heapdict[key]

    extracted_item = heapdict.popitem()
    check_heapdict_invariants(heapdict)

    with heapdict_not_changes(heapdict):
        assert key not in heapdict
        with pytest.raises(KeyError):
            _ = heapdict[key]

    assert extracted_item == (key, priority)


@given(alist=st.lists(st.integers()))
def test_sorting_by_pop_min(alist: list[int]) -> None:
    heapdict: DepqDict[int, int] = DepqDict(enumerate(alist))

    sorted_by_heapdict = []
    while heapdict:
        sorted_by_heapdict.append(heapdict.pop_min_item()[1])
        check_heapdict_invariants(heapdict)

    assert sorted_by_heapdict == sorted(alist)
    assert_heapdict_is_empty(heapdict)


@given(alist=st.lists(st.integers()))
def test_sorting_by_pop_max(alist: list[int]) -> None:
    heapdict: DepqDict[int, int] = DepqDict(enumerate(alist))

    sorted_by_heapdict = []
    while heapdict:
        sorted_by_heapdict.append(heapdict.pop_max_item()[1])
        check_heapdict_invariants(heapdict)

    assert sorted_by_heapdict == sorted(alist, reverse=True)
    assert_heapdict_is_empty(heapdict)


@given(alist=st.lists(st.integers()), random=st.randoms(use_true_random=True))
def test_sorting_by_random_extractions(alist: list[int], random) -> None:
    operations = random.choices(["pop_min", "pop_max"], k=len(alist))
    hypothesis.note(f"{operations = }")

    heapdict: DepqDict[int, int] = DepqDict(enumerate(alist))

    minimums, maximums = [], []
    for operation in operations:
        if operation == "pop_min":
            minimums.append(heapdict.pop_min_item()[1])
        elif operation == "pop_max":
            maximums.append(heapdict.pop_max_item()[1])
        check_heapdict_invariants(heapdict)
    sorted_by_heapdict = minimums + maximums[::-1]

    assert sorted_by_heapdict == sorted(alist)
    assert_heapdict_is_empty(heapdict)


@given(
    operations=st.lists(
        st.one_of(
            st.tuples(
                st.just("set"),
                st.sampled_from(range(5)),
                st.integers(),
            ),
            st.tuples(st.just("pop"), st.sampled_from(range(5))),
        )
    )
)
def test_compare_with_builtin_dict_behavior(operations: list[tuple[str, int]]) -> None:
    operation_types = {
        "set": lambda key, value: lambda d: operator.setitem(d, key, value),
        "pop": lambda key: lambda d: d.pop(key),
    }

    heapdict: DepqDict = DepqDict()
    builtin_dict: dict = {}

    for operation in operations:
        func = operation_types[operation[0]](*operation[1:])  # type: ignore[operator]
        assert_args_are_equivalent_by_function(func, heapdict, builtin_dict)
        check_heapdict_invariants(heapdict)

    assert dict(heapdict) == dict(builtin_dict)


def test_missing_key() -> None:
    heapdict = DepqDict({"a": 5, "b": 10, "c": 12})

    with heapdict_not_changes(heapdict), pytest.raises(KeyError):
        _ = heapdict["x"]

    with heapdict_not_changes(heapdict):
        assert heapdict.get("x", default=42) == 42

    with heapdict_not_changes(heapdict), pytest.raises(KeyError):
        del heapdict["x"]

    with heapdict_not_changes(heapdict), pytest.raises(KeyError):
        _ = heapdict.pop("x")

    with heapdict_not_changes(heapdict):
        assert heapdict.pop("x", 42) == 42


def test_unhashable_key() -> None:
    heapdict = DepqDict({"a": 5, "b": 10, "c": 12})

    with heapdict_not_changes(heapdict), pytest.raises(TypeError):
        heapdict[[]] = 7  # type: ignore[index]


def test_pairs_order_does_not_matter_for_equality() -> None:
    heapdict1 = DepqDict({"a": 1, "b": 2})
    heapdict2 = DepqDict({"b": 2, "a": 1})

    assert heapdict1 == heapdict2
    assert dict(heapdict1) == dict(heapdict2)
    assert OrderedDict(heapdict1) != OrderedDict(heapdict2)


def test_equals_but_nonidentical_keys_behavior() -> None:
    # Insert pairs with equal but not identical keys.
    pairs = [((1,), 3), ((1,), 1), ((1,), 2)]
    heapdict: DepqDict = DepqDict(pairs)
    heapdict[1,] = 4
    heapdict[1,] = 2
    heapdict[1,] = 5
    check_heapdict_invariants(heapdict)

    # Only one pair is stored (first inserted key and last updated priority).
    assert OrderedDict(heapdict) == OrderedDict({(1,): 5})
    assert heapdict.min_item()[0] is pairs[0][0]

    # Key is available to pop.
    heapdict.pop((1,))

    assert_heapdict_is_empty(heapdict)


def test_preserves_insertion_order_on_update() -> None:
    heapdict = DepqDict({"a": 1, "b": 5, "c": 10})
    heapdict["b"] = 20

    assert OrderedDict(heapdict) == OrderedDict({"a": 1, "b": 20, "c": 10})

    heapdict = DepqDict({"a": 1, "b": 5, "c": 10})
    del heapdict["b"]
    heapdict["b"] = 20

    assert OrderedDict(heapdict) == OrderedDict({"a": 1, "c": 10, "b": 20})


def test_clear() -> None:
    heapdict: DepqDict = DepqDict([("a", 1), ("b", 2), ("c", 3), ("d", 2), ("c", 3)])

    heapdict.clear()

    assert_heapdict_is_empty(heapdict)


def test_repr() -> None:
    heapdict: DepqDict = DepqDict()

    assert repr(heapdict) == "DepqDict()"

    heapdict = DepqDict({"a": 1, "b": 2, "c": 3})

    assert repr(heapdict) == "DepqDict({'a': 1, 'b': 2, 'c': 3})"
