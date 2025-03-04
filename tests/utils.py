from depqdict import DepqDict


def check_heapdict_invariants(heapdict: DepqDict) -> None:
    mapping = heapdict._mapping
    heap = heapdict._heap

    assert len(heapdict) == len(mapping) == len(heap)
    assert all(wrapper.key == key for key, wrapper in mapping.items())
    assert all(wrapper.index == i for i, wrapper in enumerate(heap))

    for i in range(1, len(heap)):
        parent = heapdict._get_parent(i)
        if heapdict._get_level(i) % 2 == 0:
            assert heap[parent].priority >= heap[i].priority
        else:
            assert heap[parent].priority <= heap[i].priority

    for i in range(3, len(heap)):
        grandparent = heapdict._get_grandparent(i)
        if heapdict._get_level(i) % 2 == 0:
            assert heap[grandparent].priority <= heap[i].priority
        else:
            assert heap[grandparent].priority >= heap[i].priority


def assert_heapdict_is_empty(heapdict: DepqDict) -> None:
    assert not len(heapdict)
    assert not len(heapdict._mapping)
    assert not len(heapdict._heap)
