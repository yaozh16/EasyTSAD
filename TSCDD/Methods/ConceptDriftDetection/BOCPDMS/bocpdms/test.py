import csv
import datetime
import matplotlib
import matplotlib.pylab as plt
import matplotlib.dates as mdates
import numpy as np
import os
import pytest
import sys


from TSCDD.Methods.ConceptDriftDetection.BOCPDMS.bocpdms.cp_probability_model import CpModel
from TSCDD.Methods.ConceptDriftDetection.BOCPDMS.bocpdms.detector import Detector
from TSCDD.Methods.ConceptDriftDetection.BOCPDMS.bocpdms.BVAR_NIG import BVARNIG


nile_file = os.path.join("nile.txt")
raw_data = []
count = 0
with open(nile_file) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        raw_data += row

raw_data_float = []
for entry in raw_data:
    raw_data_float.append(float(entry))
raw_data = raw_data_float

T = int(len(raw_data) / 2)
S1, S2 = 1, 1
data = np.array(raw_data).reshape(T, 2)
dates = data[:, 0]
river_height = data[:, 1]
mean, variance = np.mean(river_height), np.var(river_height)
river_height = (river_height - mean) / np.sqrt(variance)

river_height = np.array(river_height).reshape((-1, S1, S2))

"""STEP 3: Get dates"""
all_dates = []
for i in range(622+2, 1285):
    all_dates.append(datetime.date(i, 1,1))

intensity = 100
cp_model = CpModel(intensity)
a, b = 1, 1
prior_mean_scale, prior_var_scale = 0, 0.075

upper_AR = 3
lower_AR = 1
AR_models = []

for lag in range(lower_AR, upper_AR + 1):
    """Generate next model object"""
    AR_models += [BVARNIG(
        prior_a=a, prior_b=b,
        S1=S1, S2=S2,
        prior_mean_scale=prior_mean_scale,
        prior_var_scale=prior_var_scale,
        intercept_grouping=None,
        nbh_sequence=[0] * lag,
        restriction_sequence=[0] * lag,
        hyperparameter_optimization="online")]


model_universe = np.array(AR_models)
model_prior = np.array([1 / len(model_universe)] * len(model_universe))

nT = 30
detector = Detector(
    data=None,
    model_universe=model_universe,
    model_prior=model_prior,
    cp_model=cp_model,
    S1=S1, S2=S2, T=nT,
    store_rl=True, store_mrl=True,
    trim_type="keep_K", threshold=50,
    notifications=50,
    save_performance_indicators=True,
    generalized_bayes_rld="kullback_leibler",
    alpha_param_learning="individual",
    alpha_param=0.01,
    alpha_param_opt_t=30,
    alpha_rld_learning=True,
    loss_der_rld_learning="squared_loss",
    loss_param_learning="squared_loss")
start = 1
stop = T

for t in range(start-1, stop-1):
    detector.next_run(river_height[t, :], t+1)
    print(f"[{t}] {detector.CPs[t]}")