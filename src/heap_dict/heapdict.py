import copy
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from functools import partial
from typing import Generic, Iterable, Iterator, TypeVar

K = TypeVar("K")  # Type variable for keys
V = TypeVar("V")  # Type variable for values

HeapItem = tuple[K, V]


@dataclass(slots=True, order=True)
class _InternalHeapItem(Generic[K, V]):
    """Internal representation of heap item."""

    priority: V
    key: K
    index: int


# TODO add pushpop method https://docs.python.org/3/library/heapq.html#heapq.heappushpop
# TODO add double ended priority queue features https://github.com/nanouasyn/heapdict


class HeapDict(MutableMapping, Generic[K, V]):
    """Priority queue that supports retrieving and extraction keys with the
    lowest/highest priority and changing priorities for arbitrary keys.

    Implements the ``dict`` interface, the keys of which are priority queue
    elements, and the values are priorities of these elements. All keys must be
    hashable, and all values must be comparable to each other. The preservation
    of the insertion order is guaranteed in the same way as it is guaranteed
    for built-in dictionaries.
    """

    _heap: list[_InternalHeapItem[K, V]]
    _mapping: dict[K, _InternalHeapItem[K, V]]

    __slots__ = ("_heap", "_mapping")

    def __init__(
        self, iterable: None | Iterable[HeapItem] | Mapping[K, V] = None
    ) -> None:
        """Initialize priority queue instance.

        Optional *iterable* argument provides an initial iterable of pairs
        (key, priority) or {key: priority} mapping to initialize the priority
        queue.

        Other optional keyword arguments will be added in a queue as pairs:
        their names will be interpreted as keys, and their values will be
        interpreted as priorities.

        If there are several pairs with the same keys, only the last one will
        be included in the dictionary.

        >>> heapdict = HeapDict([('a', 1), ('b', 2), ('a', 3)], b=4, c=5)
        HeapDict({'a': 3, 'b': 4, 'c': 5})

        Runtime complexity: `O(n)`.
        """

        self._heap = []
        self._mapping = {}

        if iterable is None:
            return
        elif isinstance(iterable, Mapping):
            iterable_items: Iterable[HeapItem] = iterable.items()
        elif isinstance(iterable, Iterable):
            iterable_items: Iterable[HeapItem] = iterable  # type: ignore[no-redef]
        elif not isinstance(iterable, Iterable):
            raise TypeError(f"{type(iterable).__qualname__!r} object is not iterable")

        for i, (key, priority) in enumerate(iterable_items):
            if key in self._mapping:
                self._mapping[key].priority = priority
            else:
                wrapper = _InternalHeapItem(priority, key, i)
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

    def _get_selector(self, level: int):
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

    def min_item(self) -> HeapItem:
        """Return (key, priority) pair with the lowest priority.

        >>> heapdict = HeapDict({'a': 10, 'b': 5, 'c': 7})
        >>> heapdict.min_item()
        ('b', 5)

        The *default* keyword-only argument specifies an object to return if
        the dict is empty. If the dict is empty but *default* is not specified,
        a ``ValueError`` will be thrown.

        Runtime complexity: `O(1)`.
        """
        item = self._heap[0]
        return item.key, item.priority

    def pop_min_item(self) -> HeapItem:
        """Remove and return (key, priority) pair with the lowest priority.

        >>> heapdict = HeapDict({'a': 10, 'b': 5, 'c': 7})
        >>> heapdict.pop_min_item()
        ('b', 5)
        >>> heapdict
        HeapDict({'a': 10, 'c': 7})

        The *default* keyword-only argument specifies an object to return if
        the dict is empty. If the dict is empty but *default* is not specified,
        a ``ValueError`` will be thrown.

        Runtime complexity: `O(log(n))`.
        """
        item = self._heap[0]
        del self[item.key]
        return item.key, item.priority

    def max_item(self) -> HeapItem:
        """Return (key, priority) pair with the highest priority.

        >>> heapdict = HeapDict({'a': 10, 'b': 5, 'c': 7})
        >>> heapdict.max_item()
        ('a', 10)

        The *default* keyword-only argument specifies an object to return if
        the dict is empty. If the dict is empty but *default* is not specified,
        a ``ValueError`` will be thrown.

        Runtime complexity: `O(1)`.
        """
        item = self._heap[self._get_max_index()]
        return item.key, item.priority

    def pop_max_item(self) -> HeapItem:
        """Remove and return (key, priority) pair with the highest priority.

        >>> heapdict = HeapDict({'a': 10, 'b': 5, 'c': 7})
        >>> heapdict.pop_max_item()
        ('a', 10)
        >>> heapdict
        HeapDict({'b': 5, 'c': 7})

        The *default* keyword-only argument specifies an object to return if
        the dict is empty. If the dict is empty but *default* is not specified,
        a ``ValueError`` will be thrown.

        Runtime complexity: `O(log(n))`.
        """
        item = self._heap[self._get_max_index()]
        del self[item.key]
        return item.key, item.priority

    def __getitem__(self, key: K) -> V:
        """Return priority of *key*.

        >>> heapdict = HeapDict({'a': 10, 'b': 5, 'c': 7})
        >>> heapdict['a']
        10
        >>> heapdict['b']
        5

        Raises ``KeyError`` if *key* is not in the dictionary.

        RuntimeComplexity: `O(1)`.
        """
        return self._mapping[key].priority

    def __setitem__(self, key: K, priority: V) -> None:
        """Insert *key* with a specified *priority* if *key* is not in the
        dictionary, or change priority of existing *key* to *priority*
        otherwise.

        >>> heapdict = HeapDict({'a': 10, 'b': 5, 'c': 7})
        >>> heapdict['d'] = 20
        >>> heapdict['a'] = 0
        >>> heapdict
        HeapDict({'a': 0, 'b': 5, 'c': 7, 'd': 20})

        RuntimeComplexity: `O(log(n))`.
        """

        if key in self._mapping:
            self._mapping[key].priority = priority
            i = self._mapping[key].index
            self._push_up(i)
            self._push_down(i)
        else:
            wrapper = _InternalHeapItem(priority, key, len(self._heap))
            self._heap.append(wrapper)
            self._mapping[key] = wrapper
            self._push_up(len(self._heap) - 1)

    def __delitem__(self, key: K) -> None:
        """Remove *key* from the dictionary.

        >>> heapdict = HeapDict({'a': 10, 'b': 5, 'c': 7})
        >>> del heapdict['b']
        >>> heapdict
        HeapDict({'a': 10, 'c': 7})

        Raises ``KeyValue`` if *key* is not in the dictionary.

        RuntimeComplexity: `O(log(n))`.
        """
        i = self._mapping[key].index
        self._mapping.pop(key)
        end_wrapper = self._heap.pop()
        if i < len(self._heap):
            end_wrapper.index = i
            self._heap[i] = end_wrapper
            self._push_up(i)
            self._push_down(i)

    def popitem(self) -> HeapItem:
        """Remove and return a (key, priority) pair inserted last as a 2-tuple.

        Raises ``ValueError`` if dictionary is empty.

        Runtime complexity: `O(log(n))`.
        """
        if not self:
            raise ValueError("collection is empty")
        key = next(reversed(self._mapping))
        priority = self.pop(key)
        return key, priority

    def __len__(self) -> int:
        """Return the number of keys.

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

    def copy(self) -> "HeapDict[K, V]":
        """Return a shallow copy of dict."""
        heapdict = type(self)()

        for wrapper in self._heap:
            wrapper_copy = copy.copy(wrapper)
            heapdict._heap.append(wrapper_copy)
            heapdict._mapping[wrapper_copy.key] = wrapper_copy

        return heapdict

    def __copy__(self) -> "HeapDict[K, V]":
        """Return a shallow copy of dict."""
        return self.copy()

    def __repr__(self) -> str:
        """Return repr(self)."""
        if not self:
            return f"{type(self).__name__}()"

        items_str = (
            "{" + ", ".join(f"{v.key!r}: {v.priority!r}" for v in self._heap) + "}"
        )
        return f"{type(self).__name__}({items_str})"
