from collections.abc import MutableMapping
from typing import Generic, Iterable, Iterator, TypeVar, cast

K = TypeVar("K")  # Type variable for keys
V = TypeVar("V")  # Type variable for values

HeapItem = tuple[K, V]
_InternalHeapItem = list[V | K | int]


class HeapDict(MutableMapping, Generic[K, V]):
    heap: list[_InternalHeapItem]
    h_dict: dict[K, list[K | V | int]]

    def __init__(self, iterable: Iterable[HeapItem]) -> None:
        self.heap = []
        self.h_dict = {}

        for i, (key, value) in enumerate(iterable):
            wrapper = [value, key, i]
            self.heap.append(wrapper)
            self.h_dict[key] = wrapper

        n = len(self.heap)

        # Transform bottom-up.  The largest index there's any point to looking at
        # is the largest with a child index in-range, so must have 2*i + 1 < n,
        # or i < (n-1)/2.  If n is even = 2*j, this is (2*j-1)/2 = j-1/2 so
        # j-1 is the largest, which is n//2 - 1.  If n is odd = 2*j+1, this is
        # (2*j+1-1)/2 = j so j-1 is the largest, and that's again n//2-1.
        for i in range(n // 2, -1, -1):
            self._min_heapify(i)

    def clear(self) -> None:
        self.heap.clear()
        self.h_dict.clear()

    def __setitem__(self, key: K, value: V) -> None:
        if key in self.h_dict:
            self.pop(key)
        wrapper = cast(_InternalHeapItem, [value, key, len(self)])
        self.h_dict[key] = wrapper
        self.heap.append(wrapper)
        self._decrease_key(len(self.heap) - 1)

    def _min_heapify(self, i: int) -> None:
        h = self.heap
        n = len(h)

        while True:
            # calculate the offset of the left child
            left_child_pos = (i << 1) + 1
            # calculate the offset of the right child
            right_child_pos = (i + 1) << 1
            if left_child_pos < n and h[left_child_pos][0] < h[i][0]:
                low = left_child_pos
            else:
                low = i
            if right_child_pos < n and h[right_child_pos][0] < h[low][0]:
                low = right_child_pos

            if low == i:
                break

            self._swap(i, low)
            i = low

    def _decrease_key(self, i: int) -> None:
        while i:
            # calculate the offset of the parent
            parent = (i - 1) >> 1
            if self.heap[parent][0] < self.heap[i][0]:
                break
            self._swap(i, parent)
            i = parent

    def _swap(self, i: int, j: int) -> None:
        h = self.heap
        h[i], h[j] = h[j], h[i]
        h[i][2] = i
        h[j][2] = j

    def __delitem__(self, key: K) -> None:
        wrapper = self.h_dict[key]
        while wrapper[2]:
            # calculate the offset of the parent
            parentpos = (cast(int, wrapper[2]) - 1) >> 1
            parent = self.heap[parentpos]
            self._swap(cast(int, wrapper[2]), parent[2])
        self.popitem()

    def __getitem__(self, key: K) -> V:
        return cast(V, self.h_dict[key][0])

    def __iter__(self) -> Iterator[K]:
        return iter(self.h_dict)

    def popitem(self) -> HeapItem:
        """D.popitem() -> (k, v), remove and return the (key, value) pair with lowest\nvalue; but raise KeyError if D is empty."""
        wrapper = self.heap[0]
        if len(self.heap) == 1:
            self.heap.pop()
        else:
            self.heap[0] = self.heap.pop()
            self.heap[0][2] = 0
            self._min_heapify(0)

        value, key, _ = wrapper
        self.h_dict.pop(cast(K, key))
        return key, value

    def __len__(self) -> int:
        return len(self.h_dict)

    def peekitem(self) -> HeapItem:
        """
        D.peekitem() -> (k, v), return the (key, value) pair with lowest value;
        but raise KeyError if D is empty.
        """
        value, key, _ = self.heap[0]
        return key, value
