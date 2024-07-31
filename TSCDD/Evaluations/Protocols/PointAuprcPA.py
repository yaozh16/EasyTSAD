from typing import Type
from .. import MetricInterface, EvalInterface
from ..Metrics import SingleValueMetric
from ..utils import rec_scores
import sklearn.metrics
import math
from matplotlib import pyplot as plt

class PointAuprcPA(EvalInterface):
    """
    Using Point-based point-adjustment Auprc to evaluate the models.
    """
    def __init__(self) -> None:
        super().__init__("point-based auprc pa")
        self.figname = None
        
    def calc(self, scores, labels, margins) -> MetricInterface:
        '''
        Returns:
         An Auprc instance (Evaluations.Metrics.Auprc), including:\n
            auprc: auprc value.
        '''
        scores = rec_scores(scores=scores, labels=labels)
        auprc = sklearn.metrics.average_precision_score(y_true=labels, 
                                                        y_score=scores, average=None)
        
        if math.isnan(auprc):
            auprc = 0
        
        ## plot
        if self.figname:
            prec, recall, _ = sklearn.metrics.precision_recall_curve(y_true=labels,
                                                                     probas_pred=scores)
            display = sklearn.metrics.PrecisionRecallDisplay(precision=prec, 
                                                             recall=recall)
            display.plot()
            plt.savefig(str(self.figname) + "_auprc.pdf")
            
        return SingleValueMetric(value=auprc, name=self.name)