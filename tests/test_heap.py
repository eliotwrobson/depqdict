from __future__ import print_function
import random
from heap_dict import HeapDict

N = 100


import random

class TestHeap:
    def check_invariants(self, h):
        for i, e in enumerate(h.heap):
            assert e[2] == i
        for i in range(1, len(h.heap)):
            parent = (i - 1) >> 1
            assert h.heap[parent][0] <= h.heap[i][0]

    def make_data(self):
        pairs = [(random.random(), random.random()) for _ in range(N)]
        h = HeapDict()
        d = {k: v for k, v in pairs}
        pairs.sort(key=lambda x: x[1], reverse=True)
        return h, pairs, d

    def test_popitem(self):
        h, pairs, _ = self.make_data()
        while pairs:
            v = h.popitem()
            v2 = pairs.pop(-1)
            assert v == v2
        assert len(h) == 0

    def test_popitem_ties(self):
        h = HeapDict()
        for i in range(N):
            h[i] = 0
        for _ in range(N):
            _, v = h.popitem()
            assert v == 0
            self.check_invariants(h)

    def test_peek(self):
        h, pairs, _ = self.make_data()
        while pairs:
            v = h.peekitem()[0]
            h.popitem()
            v2 = pairs.pop(-1)
            assert v == v2[0]
        assert len(h) == 0

    def test_iter(self):
        h, _, d = self.make_data()
        assert list(h) == list(d)

    def test_keys(self):
        h, _, d = self.make_data()
        assert sorted(h.keys()) == sorted(d.keys())

    def test_values(self):
        h, _, d = self.make_data()
        assert sorted(h.values()) == sorted(d.values())

    def test_del(self):
        h, pairs, _ = self.make_data()
        k, _ = pairs.pop(N // 2)
        del h[k]
        while pairs:
            v = h.popitem()
            v2 = pairs.pop(-1)
            assert v == v2
        assert len(h) == 0

    def test_change(self):
        h, pairs, _ = self.make_data()
        k, _ = pairs[N // 2]
        h[k] = 0.5
        pairs[N // 2] = (k, 0.5)
        pairs.sort(key=lambda x: x[1], reverse=True)
        while pairs:
            v = h.popitem()
            v2 = pairs.pop(-1)
            assert v == v2
        assert len(h) == 0

    def test_clear(self):
        h, _, _ = self.make_data()
        h.clear()
        assert len(h) == 0
