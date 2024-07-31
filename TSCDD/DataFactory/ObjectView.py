from abc import ABC, abstractmethod
from itertools import accumulate
import numpy as np
from typing import Union


class ObjectView(ABC):
    def __init__(self, obj, start_index: int, end_index: int, safety_check: bool = False):
        self._obj = obj
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = len(obj)
        if safety_check:
            last = end_index if end_index >= 0 else len(obj) + end_index
            last = int(min(last, len(obj)))
            first = start_index if start_index >= 0 else len(obj) + start_index
            last = int(max(first, last))
            self.start_index = first
            self.end_index = last
        else:
            self.start_index = start_index
            self.end_index = end_index

    def size(self) -> int:
        return self.end_index - self.start_index

    def __len__(self) -> int:
        return self.size()

    def split_by_fractions(self, fractions: Union[float, list[float]]):
        if isinstance(fractions, float):
            fractions = min(max(fractions, 0.0), 1.0)
            fractions = [fractions, 1 - fractions]
        fractions = np.array(fractions) / sum(fractions) * self.size()
        glb_fractions = [0] + list(accumulate(fractions))[:-1] + [self.size()]
        views = []
        for start_index, end_index in zip(glb_fractions[:-1], glb_fractions[1:]):
            seg = self.__class__(self._obj, int(start_index + self.start_index), int(end_index + self.start_index))
            seg._inherit(self)
            views.append(seg)
        return views

    def head(self, n: int, safety_check=True):
        assert isinstance(n, int) and n > 0, f"ObjectView head parameter should be positive integer " \
                                             f"(currently:[{type(n)}]{n})"
        if safety_check:
            if n >= self.size():
                n = self.size()
        seg = self.__class__(self._obj, self.start_index, self.start_index + n)
        seg._inherit(self)
        return seg

    def last(self, n: int, safety_check=True):
        assert isinstance(n, int) and n > 0, "ObjectView last parameter should be positive integer"
        if safety_check:
            if n >= self.size():
                n = self.size()
        seg = self.__class__(self._obj, self.end_index - n, self.end_index)
        seg._inherit(self)
        return seg

    def safe_slice(self, start_index: int, end_index: int):
        start = start_index + self.size() if start_index < 0 else start_index
        end = end_index + self.size() if end_index < 0 else end_index
        assert start >= 0 and end >= 0, f"Invalid slice index: {start_index},{end_index}"
        assert start <= end, f"Invalid slice index: {start_index}({start})>={end_index}({end})"
        if start < 0:
            start = 0
        if end > self.size():
            end = self.size()
        seg = self.__class__(self._obj, start + self.start_index, end + self.start_index, True)
        seg._inherit(self)
        return seg

    @abstractmethod
    def _inherit(self, parent: "ObjectView"):
        raise NotImplementedError(f"{self.__class__} _inherit is not implemented.")

    def crop_size(self, size: int, align_right: bool = True):
        assert size > 0, "ObjectView crop size must be positive"
        if size >= self.size():
            return
        if align_right:
            self.start_index = self.end_index - size
        else:
            self.end_index = self.start_index + size
