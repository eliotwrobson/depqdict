# Based on tests from heapdict by Daniel Stutzbach
# https://github.com/DanielStutzbach/heapdict/blob/master/test_heap.py

import heapq
import random

from heap_dict import HeapDict

SEED = hash("Tarjan")
NUM_RANGE = 100_000_000


def check_invariants(h) -> None:
    for i, e in enumerate(h._heap):
        assert e.index == i
    for i in range(1, len(h._heap)):
        parent = (i - 1) >> 1
        assert h._heap[parent].index <= h._heap[i].index


def make_data(
    n: int,
) -> tuple[HeapDict[int, float], list[tuple[int, float]], dict[int, float]]:
    random.seed(SEED)
    pairs: list[tuple[int, float]] = [
        (
            random.randint(0, NUM_RANGE),
            random.randint(0, NUM_RANGE),
        )
        for _ in range(n)
    ]
    h: HeapDict[int, float] = HeapDict(pairs)
    d = dict(pairs)
    pairs.sort(key=lambda x: x[1], reverse=True)
    return h, pairs, d


def test_popitem() -> None:
    h, pairs, _ = make_data(100)

    while pairs:
        v = h.pop_min_item()
        v2 = pairs.pop(-1)
        assert v == v2
    assert len(h) == 0


def test_popitem_ties() -> None:
    h: HeapDict[int, float] = HeapDict()
    for i in range(100):
        h[i] = 0
    for _ in range(100):
        _, v = h.pop_min_item()
        assert v == 0
        check_invariants(h)


def test_peek() -> None:
    h, pairs, _ = make_data(100)
    while pairs:
        v = h.min_item()[0]
        h.pop_min_item()
        v2 = pairs.pop(-1)
        assert v == v2[0]
    assert len(h) == 0


def test_iter() -> None:
    h, _, d = make_data(100)
    assert set(h) == set(d.keys())


def test_keys() -> None:
    h, _, d = make_data(100)
    assert sorted(h.keys()) == sorted(d.keys())


def test_values() -> None:
    h, _, d = make_data(100)
    assert set(d.values()) == set(h.values())


def test_del() -> None:
    h, pairs, _ = make_data(100)
    k, _ = pairs.pop(50)
    del h[k]
    while pairs:
        v = h.pop_min_item()
        v2 = pairs.pop(-1)
        assert v == v2
    assert len(h) == 0


def test_change() -> None:
    h, pairs, _ = make_data(100)
    k, _ = pairs[50]
    h[k] = 0.5
    pairs[50] = (k, 0.5)
    pairs.sort(key=lambda x: x[1], reverse=True)
    while pairs:
        v = h.pop_min_item()
        v2 = pairs.pop(-1)
        assert v == v2
    assert len(h) == 0


def test_clear() -> None:
    h, _, _ = make_data(100)
    h.clear()
    assert len(h) == 0


def test_max_k_items() -> None:
    n = 10_000
    k = 100
    _, d, _ = make_data(n)

    random.shuffle(d)

    new_heap: HeapDict[int, float] = HeapDict()

    for key, value in d:
        if len(new_heap) < k:
            new_heap[key] = value
        else:
            thing = new_heap.push_pop_min_item(key, value)
            assert thing[1] <= new_heap.min_item()[1]

    expected_result = heapq.nlargest(k, d, key=lambda x: x[1])
    assert len(new_heap) == len(expected_result)

    res = []
    while new_heap:
        res.append(new_heap.pop_max_item())

    assert res == expected_result
    assert len(new_heap) == 0


def test_min_k_items() -> None:
    n = 10_000
    k = 100
    _, d, _ = make_data(n)

    random.shuffle(d)

    new_heap: HeapDict[int, float] = HeapDict()

    for key, value in d:
        if len(new_heap) < k:
            new_heap[key] = value
        else:
            thing = new_heap.push_pop_max_item(key, value)
            assert thing[1] >= new_heap.max_item()[1]

    expected_result = heapq.nsmallest(k, d, key=lambda x: x[1])
    assert len(new_heap) == len(expected_result)

    res = []
    while new_heap:
        res.append(new_heap.pop_min_item())

    assert res == expected_result
    assert len(new_heap) == 0
