import numpy as np
import logging
from typing import Union, Tuple
from . import MetricInterface, EvalInterface
from ..DataFactory.LabelStore import LabelView
from .EvalPreprocess import EvalPreprocess, EvalPreprocessOption


logger = logging.getLogger("logger")


class Performance:
    def __init__(self, method_output: LabelView, ground_truth: LabelView, margins) -> None:
        '''
        Init Performance and check if the format of scores is valid. 
        
        Notice:
         If the length of the scores is less than labels, 
         then labels are cut to labels[len(labels) - len(scores):]
         
        Params:
         - `scores` - the anomaly scores provided by methods\n
         - `labels` - the ground truth labels\n
        '''

        mo: np.ndarray = method_output.get()
        gt: np.ndarray = ground_truth.get()
        self.all_label_normal = np.all(gt < 0.5)
        if self.all_label_normal:
            return
        
        self.margins = margins
        
        try:
            if not isinstance(mo, np.ndarray):
                raise TypeError("Invalid scores type. Make sure that scores are np.ndarray\n")
            if not isinstance(gt, np.ndarray):
                raise TypeError("Invalid label type. Make sure that labels are np.ndarray\n")
            if mo.ndim != 1:
                raise ValueError("Invalid scores dimension, the dimension must be 1.\n")
            if len(mo) > len(gt):
                raise AssertionError("Score length must less than label length! Score length: {}; Label length: {}".
                                     format(len(mo), len(gt)))
        except Exception as e:
            raise e

        self.eval_preprocessor: EvalPreprocess = EvalPreprocess(method_output, ground_truth, margins)

    def perform_eval(self, evaluators: list[EvalInterface]) -> Union[None, Tuple[list[MetricInterface], dict]]:
        if self.all_label_normal:
            return None
        metrics: list[MetricInterface] = []
        for evaluator in evaluators:
            if not isinstance(evaluator, EvalInterface):
                raise TypeError(f"Please set EvalInterface objects (currently {evaluator.__class__}).")
            scores, labels = self.eval_preprocessor.get_eval_input(evaluator.preprocess_option)
            item_result: MetricInterface = evaluator.calc(scores, labels, self.margins)
            if not isinstance(item_result, MetricInterface):
                raise TypeError("Return value of func 'calc' must be inherented from MetricInterface.")
            metrics.append(item_result)
        
        res_dict = MetricInterface.list_to_dict(metrics)
            
        return metrics, res_dict

