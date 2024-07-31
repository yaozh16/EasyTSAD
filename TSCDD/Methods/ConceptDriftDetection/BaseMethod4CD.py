from abc import ABC, abstractmethod
from ..BaseMethods import BaseMethod, BaseOfflineMethod, BaseOnlineMethod
from ...DataFactory.DataStore import DataStoreViewIndex


class BaseMethod4CD(BaseMethod, ABC):

    def __init__(self, hparams: dict = None):
        super().__init__(hparams)

    def train_valid(self, data_store_view_index: DataStoreViewIndex):
        pass


class BaseOfflineMethod4CD(BaseMethod4CD, BaseOfflineMethod, ABC):
    pass


class BaseOnlineMethod4CD(BaseMethod4CD, BaseOnlineMethod, ABC):
    pass



