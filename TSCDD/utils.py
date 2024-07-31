from collections.abc import Iterable, Sized
from tqdm import tqdm
from collections import OrderedDict
from datetime import datetime
from logging import Logger
from typing import Union


def update_nested_dict(d1, d2) -> dict:
    for k, v in d2.items():
        if isinstance(v, dict):
            d1[k] = update_nested_dict(d1.get(k, {}), v)
        else:
            d1[k] = v
    return d1


class Progress(Iterable, Sized):

    def __init__(self, iterable_obj, logger: Union[Logger, str] = None, log_level=None):
        self._obj = iterable_obj
        self._logger = logger
        self._log_level = log_level
        self._desc = None
        self._msgs = OrderedDict()

    def __iter__(self):
        self._desc = tqdm(self._obj)
        for e in self._desc:
            yield e

    def __len__(self):
        return len(self._obj)

    def info(self, msg, keyword="Default"):
        if keyword in self._msgs and self._msgs[keyword] == msg:
            return
        self._msgs[keyword] = msg
        self._update_msg()

    def _update_msg(self):
        if self._desc is None:
            return
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        log_prefix = f"({current_time}) [Progress] "
        log_suffix = " ".join([v for k, v in self._msgs.items()])
        self._desc.set_description(f"{log_prefix} {log_suffix}")
        if isinstance(self._logger, Logger):
            self._logger.log(level=self._log_level, msg=log_suffix)

    def flush_msg(self):
        self._msgs = OrderedDict()
