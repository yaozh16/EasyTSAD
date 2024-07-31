import numpy as np
import torch

from ....DataFactory.TimeSeries import TimeSeriesView
from .TCNEncoder import TCNEncoder
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from ..BaseMethod4CD import BaseOfflineMethod4CD, BaseOnlineMethod
from ... import MethodTestResults
from ....DataFactory.LabelStore import LabelStore, ChangePointLabel, ReportPointLabel, RunLengthLabel


class SlidingWindowDataset(Dataset):
    def __init__(self, data: torch.Tensor, window_size: int, stride: int = 1):
        # shape (n_dimension, n_observation)
        self.data = data
        self.stride = stride
        self.window_size = window_size
        self.num_samples = (data.shape[1] - window_size) // stride + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx *= self.stride
        return self.data[:, idx:idx + self.window_size]


class TSCP2Model:
    def __init__(self, num_dim: int, window_size: int, code_size: int, LR: float = 0.001, decay_steps: int = 1000,
                 temperature: float = 0.1, device="cpu"):
        self._code_size = code_size
        self._window_size = window_size
        self._num_dim = num_dim
        self._temperature = temperature
        self._device = device

        self._encoder = TCNEncoder(num_dim=num_dim, window_size=window_size, code_size=code_size).to(device)
        self._criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self._optimizer = optim.Adam(self._encoder.parameters(), lr=LR)
        self._scheduler = CosineAnnealingLR(self._optimizer, T_max=decay_steps)
        self._raw_ts: torch.Tensor = torch.zeros(size=(num_dim, window_size * 2)).to(self._device)

    def update(self, timeseries: np.ndarray):
        if timeseries.ndim == 1:
            assert timeseries.shape[0] == self._num_dim
            timeseries = timeseries.reshape((self._num_dim, 1))
        else:
            assert timeseries.ndim == 2 and timeseries.shape[1] == self._num_dim
            timeseries = timeseries.T
        D, L = timeseries.shape
        if L > 2 * self._window_size:
            self._raw_ts = torch.from_numpy(timeseries[:, -self._window_size * 2:]).float().to(self._device)
        else:
            self._raw_ts[:, :L] = torch.from_numpy(timeseries).float().to(self._device)
            self._raw_ts = torch.roll(self._raw_ts, shifts=-L, dims=1)

    def test_change(self, threshold=0.5):
        with torch.no_grad():
            # shape: (1, n_dim, window_size)
            history = self._raw_ts[:, :self._window_size].unsqueeze(0)
            future = self._raw_ts[:, -self._window_size:].unsqueeze(0)
            # shape: (1, code_size)
            history_code = self._encoder(history)
            future_code = self._encoder(future)
            similarity = self._similarity_cosine(history_code, future_code).item()
            return similarity < threshold

    def eval(self):
        self._encoder.eval()

    def train(self, timeseries: np.ndarray, epoch_num: int, stride: int = 1, batch_size: int = 4):
        N, D = timeseries.shape
        assert D == self._num_dim, f"The dimension of training timeseries ({D}) is not ({self._num_dim}) as configured."
        timeseries = torch.from_numpy(timeseries).float().t().to(self._device)
        self._encoder.train()
        dataset = SlidingWindowDataset(timeseries, window_size=self._window_size * 2, stride=stride)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_losses = []
        for i in range(epoch_num):
            epoch_losses = self._train_epoch(dataloader)
            epoch_losses = np.sum(epoch_losses, axis=0)
            train_losses.append(epoch_losses)
        train_losses = np.array(train_losses)
        return train_losses

    def _train_epoch(self, dataloader: DataLoader):
        losses = []
        for batch_data in dataloader:
            batch_history = batch_data[:, :, :self._window_size]
            batch_future = batch_data[:, :, -self._window_size:]
            loss, mean_sim, mean_neg = self._train_batch(batch_history, batch_future)
            losses.append([mean_sim.item(), mean_neg.item()])
        return np.array(losses)

    def _train_batch(self, batch_x1: torch.Tensor, batch_x2: torch.Tensor):
        batch_size = batch_x1.shape[0]
        assert batch_x1.shape == (batch_size, self._num_dim, self._window_size)
        assert batch_x2.shape == (batch_size, self._num_dim, self._window_size)

        self._optimizer.zero_grad()
        batch_code1: torch.Tensor = self._encoder(batch_x1)
        batch_code2: torch.Tensor = self._encoder(batch_x2)

        loss, mean_sim, mean_neg = self._nce_loss(batch_code1, batch_code2, temperature=self._temperature)

        loss.backward()  # 反向传播，计算当前梯度
        self._optimizer.step()  # 更新模型参数
        self._scheduler.step()  # 更新学习率
        return loss, mean_sim, mean_neg

    def _nce_loss(self, batch_code1: torch.Tensor, batch_code2: torch.Tensor, temperature=0.1):
        batch_size = batch_code1.shape[0]
        similarity_matrix: torch.Tensor = self._similarity_cosine(batch_code1, batch_code2)

        pos_sim = torch.exp(torch.diag(similarity_matrix) / temperature)

        tri_mask = torch.ones(batch_size, batch_size, dtype=torch.bool).to(self._device)
        tri_mask.fill_diagonal_(0)

        neg = similarity_matrix.masked_select(tri_mask).view(batch_size, batch_size - 1)
        all_sim = torch.exp(similarity_matrix / temperature)

        logits = pos_sim.sum() / all_sim.sum(dim=1)

        lbl = torch.ones(batch_size, dtype=torch.float).to(self._device)
        loss = self._criterion(logits, lbl)

        mean_sim = torch.diag(similarity_matrix).mean()
        mean_neg = neg.mean()
        return loss, mean_sim, mean_neg

    def _similarity_cosine(self, batch_code1: torch.Tensor, batch_code2: torch.Tensor):
        batch_size = batch_code1.shape[0]
        assert batch_code1.shape == (batch_size, self._code_size)
        assert batch_code2.shape == (batch_size, self._code_size)
        similarity_matrix = torch.mm(batch_code1, batch_code2.t())  # (batch_size, batch_size
        return similarity_matrix


class TSCP2(BaseOfflineMethod4CD):
    def offline_initialize(self):
        pass

    def offline_test(self, timeseries: TimeSeriesView) -> MethodTestResults:
        window_size = self.hparams.get("window_size", 10)
        code_size = self.hparams.get("code_size", 10)
        epoch_num = self.hparams.get("epoch_num", 100)
        stride = self.hparams.get("stride", 2)
        batch_size = self.hparams.get("batch_size", 64)
        temperature = self.hparams.get("temperature", 0.1)
        device = self.hparams.get("device", "cuda")
        lr = self.hparams.get("lr", 1e-4)
        change_threshold = self.hparams.get("change_threshold", 0.5)

        n_dim = timeseries.get_dim()
        n_obs = timeseries.size()
        if n_obs < window_size * 3:
            window_size = int(n_obs//3)

        model = TSCP2Model(num_dim=n_dim, window_size=window_size, code_size=code_size, LR=lr, temperature=temperature,
                           device=device)
        values = timeseries.get_values().reshape((-1, n_dim))
        model.train(values, epoch_num=epoch_num, stride=stride, batch_size=batch_size)
        change_points_mask = np.zeros(n_obs)
        report_points_mask = np.zeros(n_obs)
        run_lengths = np.arange(n_obs)
        model.eval()
        for i, line in enumerate(values):
            model.update(line)
            if i >= window_size * 2:
                change_mask = model.test_change(threshold=change_threshold)
                if change_mask:
                    report_points_mask[i] = 1
                    change_points_mask[i - window_size] = 1
                    run_lengths[i:] -= run_lengths[i] - window_size

        change_point_label = ChangePointLabel(change_points_mask, annotator="TSCP2(CP)")
        report_point_label = ReportPointLabel(report_points_mask, annotator="TSCP2(RP)")
        run_length_label = RunLengthLabel(run_lengths, annotator="TSCP2(RL)")

        self.test_results = MethodTestResults(LabelStore([
            change_point_label,
            report_point_label,
            run_length_label
        ]))
        return self.test_results

    @classmethod
    def _method_file_path(cls) -> str:
        return __file__
