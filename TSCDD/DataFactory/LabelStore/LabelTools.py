import numpy as np


class LabelTools:
    @staticmethod
    def convert_binary_to_run_length(binary_label: np.ndarray):
        run_length = np.zeros_like(binary_label)

        arr_ext = np.concatenate([np.ones_like(binary_label[:1]), binary_label, np.ones_like(binary_label[:1])])
        seg_starts, = np.where(arr_ext[:-1] & ~arr_ext[1:])
        seg_ends, = np.where(~arr_ext[:-1] & arr_ext[1:])
        for seg_start, seg_end in zip(seg_starts, seg_ends):
            run_length[seg_start: seg_end] = np.arange(seg_end - seg_start)
        return run_length, (seg_starts, seg_ends)


