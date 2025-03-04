# Based on heapdict by Evgeniy Selezniov, see
# https://github.com/nanouasyn/heapdict/blob/main/heapdict.py


import copy
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass
from functools import partial
from typing import Generic, TypeVar

K = TypeVar("K")  # Type variable for keys
V = TypeVar("V")  # Type variable for values

DepqItem = tuple[K, V]


@dataclass(slots=True, order=True)
class _InternalDepqItem(Generic[K, V]):
    """Internal representation of depq item."""

    priority: V
    key: K
    index: int


class DepqDict(MutableMapping, Generic[K, V]):
    """
    Priority queue that supports retrieving and extraction keys with the
    lowest/highest priority and changing priorities for arbitrary keys.

    Implements the ``dict`` interface, the keys of which are priority queue
    elements, and the values are priorities of these elements. All keys must be
    hashable, and all values must be comparable to each other.
    """

    _heap: list[_InternalDepqItem[K, V]]
    _mapping: dict[K, _InternalDepqItem[K, V]]

    __slots__ = ("_heap", "_mapping")

    def __init__(self, iterable: None | Iterable[DepqItem[K, V]] | Mapping[K, V] = None) -> None:
        """
        Initialize priority queue instance.

        Optional *iterable* argument provides an initial iterable of pairs
        (key, priority) or {key: priority} mapping to initialize the priority
        queue.

        If there are several pairs with the same keys, only the last one will
        be included in the dictionary.

        >>> depqdict = DepqDict([('a', 1), ('b', 2), ('a', 3)], b=4, c=5)
        DepqDict({'a': 3, 'b': 4, 'c': 5})

        Runtime complexity: `O(n)`.
        """

        self._heap = []
        self._mapping = {}

        if iterable is None:
            return
        elif isinstance(iterable, Mapping):
            iterable_items: Iterable[DepqItem[K, V]] = iterable.items()
        elif isinstance(iterable, Iterable):
            iterable_items: Iterable[DepqItem[K, V]] = iterable  # type: ignore[no-redef]
        elif not isinstance(iterable, Iterable):
            raise TypeError(f"{type(iterable).__qualname__!r} object is not iterable")

        for i, (key, priority) in enumerate(iterable_items):
            if key in self._mapping:
                self._mapping[key].priority = priority
            else:
                wrapper = _InternalDepqItem(priority, key, i)
                self._heap.append(wrapper)
                self._mapping[key] = wrapper

        # Restoring the heap invariant.
        push_down = self._push_down
        for i in reversed(range(len(self._heap) // 2)):
            push_down(i)

    def _swap(self, i: int, j: int) -> None:
        h = self._heap
        h[i], h[j] = h[j], h[i]
        h[i].index = i
        h[j].index = j

    def _get_level(self, i: int) -> int:
        return (i + 1).bit_length() - 1

    def _get_parent(self, i: int) -> int:
        return (i - 1) // 2

    def _get_grandparent(self, i: int) -> int:
        return (i - 3) // 4

    def _with_children(self, i: int) -> Iterable[int]:
        yield i
        first = 2 * i + 1
        yield from range(first, min(len(self._heap), first + 2))

    def _with_grandchildren(self, i: int) -> Iterable[int]:
        yield i
        first = 4 * i + 3
        yield from range(first, min(len(self._heap), first + 4))

    def _get_selector(self, level: int) -> Callable[..., int]:
        heap = self._heap
        selector = [min, max][level % 2]
        return partial(selector, key=lambda i: heap[i].priority)

    def _push_down(self, i: int) -> None:
        with_children = self._with_children
        with_grandchildren = self._with_grandchildren
        select = self._get_selector(self._get_level(i))
        while True:
            should_be_parent = select(with_children(i))
            if should_be_parent != i:
                self._swap(i, should_be_parent)

            should_be_grandparent = select(with_grandchildren(i))
            if should_be_grandparent == i:
                return
            self._swap(i, should_be_grandparent)
            i = should_be_grandparent

    def _push_up(self, i: int) -> None:
        parent = self._get_parent(i)
        if parent < 0:
            return
        select = self._get_selector(self._get_level(parent))
        if select(parent, i) == i:
            self._swap(i, parent)
            i = parent

        get_grandparent = self._get_grandparent
        select = self._get_selector(self._get_level(i))
        while (grandparent := get_grandparent(i)) >= 0:
            if select(grandparent, i) == grandparent:
                break
            self._swap(i, grandparent)
            i = grandparent

    def _get_max_index(self) -> int:
        length = len(self._heap)
        return self._get_selector(1)(1, 2) if length > 2 else length - 1

    def min_item(self) -> DepqItem[K, V]:
        """
        Return (key, priority) pair with the lowest priority.

        >>> depqdict = DepqDict({'a': 10, 'b': 5, 'c': 7})
        >>> depqdict.min_item()
        ('b', 5)

        Runtime complexity: `O(1)`.
        """
        item = self._heap[0]
        return item.key, item.priority

    def pop_min_item(self) -> DepqItem[K, V]:
        """
        Remove and return (key, priority) pair with the lowest priority.

        >>> depqdict = DepqDict({'a': 10, 'b': 5, 'c': 7})
        >>> depqdict.pop_min_item()
        ('b', 5)
        >>> depqdict
        DepqDict({'a': 10, 'c': 7})

        Runtime complexity: `O(log(n))`.
        """
        item = self._push_pop(0, None)
        return item.key, item.priority

    def push_pop_min_item(self, key: K, priority: V) -> DepqItem[K, V]:
        """
        Push item into the heap and return the smallest item. Faster than
        using push and then pop_min_item.

        >>> depqdict = DepqDict({'a': 10, 'b': 5, 'c': 7})
        >>> depqdict.push_pop_min_item(('d', 3))
        ('b', 5)
        >>> depqdict
        DepqDict({'c': 7, 'd': 3, 'a': 10})

        Runtime complexity: `O(log(n))`.
        """

        if not self._heap or (self._heap and priority < self._heap[0].priority):  # type: ignore[operator]
            return key, priority

        res = self._push_pop(0, _InternalDepqItem(priority, key, 0))
        return res.key, res.priority

    def max_item(self) -> DepqItem[K, V]:
        """
        Return (key, priority) pair with the highest priority.

        >>> depqdict = DepqDict({'a': 10, 'b': 5, 'c': 7})
        >>> depqdict.max_item()
        ('a', 10)

        The *default* keyword-only argument specifies an object to return if
        the dict is empty. If the dict is empty but *default* is not specified,
        a ``ValueError`` will be thrown.

        Runtime complexity: `O(1)`.
        """
        item = self._heap[self._get_max_index()]
        return item.key, item.priority

    def pop_max_item(self) -> DepqItem[K, V]:
        """
        Remove and return (key, priority) pair with the highest priority.

        >>> depqdict = DepqDict({'a': 10, 'b': 5, 'c': 7})
        >>> depqdict.pop_max_item()
        ('a', 10)
        >>> depqdict
        DepqDict({'b': 5, 'c': 7})

        The *default* keyword-only argument specifies an object to return if
        the dict is empty. If the dict is empty but *default* is not specified,
        a ``ValueError`` will be thrown.

        Runtime complexity: `O(log(n))`.
        """
        item = self._push_pop(self._get_max_index(), None)
        return item.key, item.priority

    def push_pop_max_item(self, key: K, priority: V) -> DepqItem[K, V]:
        """
        Push item into the heap and return the largest item. Faster than
        push and then pop_max_item.

        >>> depqdict = DepqDict({'a': 10, 'b': 5, 'c': 7})
        >>> depqdict.push_pop_max_item(('d', 3))
        ('a', 10)
        >>> depqdict
        DepqDict({'c': 7, 'd': 3, 'b': 5})

        Runtime complexity: `O(log(n))`.
        """

        max_idx = self._get_max_index()
        if not self._heap or (self._heap and priority > self._heap[max_idx].priority):  # type: ignore[operator]
            return key, priority

        res = self._push_pop(max_idx, _InternalDepqItem(priority, key, max_idx))
        return res.key, res.priority

    def _push_pop(self, existing_idx: int, new_item: None | _InternalDepqItem[K, V]) -> _InternalDepqItem[K, V]:
        old_item = self._heap[existing_idx]
        self._mapping.pop(old_item.key)

        if new_item is None:
            if existing_idx == len(self._heap) - 1:
                return self._heap.pop()

            item_to_add = self._heap.pop()
            item_to_add.index = existing_idx
        else:
            item_to_add = new_item
            self._mapping[new_item.key] = item_to_add

        self._heap[existing_idx] = item_to_add
        self._push_up(existing_idx)
        self._push_down(existing_idx)

        return old_item

    def __getitem__(self, key: K) -> V:
        """
        Return priority of *key*.

        >>> depqdict = DepqDict({'a': 10, 'b': 5, 'c': 7})
        >>> depqdict['a']
        10
        >>> depqdict['b']
        5

        Raises ``KeyError`` if *key* is not in the dictionary.

        RuntimeComplexity: `O(1)`.
        """
        return self._mapping[key].priority

    def __setitem__(self, key: K, priority: V) -> None:
        """
        Insert *key* with a specified *priority* if *key* is not in the
        dictionary, or change priority of existing *key* to *priority*
        otherwise.

        >>> depqdict = DepqDict({'a': 10, 'b': 5, 'c': 7})
        >>> depqdict['d'] = 20
        >>> depqdict['a'] = 0
        >>> depqdict
        DepqDict({'a': 0, 'b': 5, 'c': 7, 'd': 20})

        RuntimeComplexity: `O(log(n))`.
        """

        if key in self._mapping:
            item = self._mapping[key]
            item.priority = priority
            i = item.index
            self._push_up(i)
            self._push_down(i)
        else:
            wrapper = _InternalDepqItem(priority, key, len(self._heap))
            self._heap.append(wrapper)
            self._mapping[key] = wrapper
            self._push_up(wrapper.index)

    def __delitem__(self, key: K) -> None:
        """
        Remove *key* from the dictionary.

        >>> depqdict = DepqDict({'a': 10, 'b': 5, 'c': 7})
        >>> del depqdict['b']
        >>> depqdict
        DepqDict({'a': 10, 'c': 7})

        Raises ``KeyValue`` if *key* is not in the dictionary.

        RuntimeComplexity: `O(log(n))`.
        """
        item = self._mapping.pop(key)
        i = item.index
        end_wrapper = self._heap.pop()
        if i < len(self._heap):
            end_wrapper.index = i
            self._heap[i] = end_wrapper
            self._push_up(i)
            self._push_down(i)

    def popitem(self) -> DepqItem[K, V]:
        """
        Remove and return a (key, priority) pair inserted last as a 2-tuple.

        Raises ``ValueError`` if dictionary is empty.

        Runtime complexity: `O(log(n))`.
        """
        if not self:
            raise ValueError("collection is empty")
        key = next(reversed(self._mapping))
        priority = self.pop(key)
        return key, priority

    def __len__(self) -> int:
        """
        Return the number of keys.

        Runtime complexity: `O(1)`
        """
        return len(self._heap)

    def __iter__(self) -> Iterator[K]:
        """Return keys iterator."""
        return iter(self._mapping)

    def clear(self) -> None:
        """Remove all items from dict."""
        self._heap.clear()
        self._mapping.clear()

    def copy(self) -> "DepqDict[K, V]":
        """Return a shallow copy of dict."""
        depqdict = type(self)()

        for wrapper in self._heap:
            wrapper_copy = copy.copy(wrapper)
            depqdict._heap.append(wrapper_copy)
            depqdict._mapping[wrapper_copy.key] = wrapper_copy

        return depqdict

    def __copy__(self) -> "DepqDict[K, V]":
        """Return a shallow copy of dict."""
        return self.copy()

    def __repr__(self) -> str:
        """Return repr(self)."""
        if not self:
            return f"{type(self).__name__}()"

        items_str = "{" + ", ".join(f"{v.key!r}: {v.priority!r}" for v in self._heap) + "}"
        return f"{type(self).__name__}({items_str})"
