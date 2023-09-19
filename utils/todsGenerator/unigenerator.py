'''
 Licensed under the Apache License, Version 2.0 (the "License");
 ©Copyright [2021] [DataLab@Rice University]
 
 Source code from https://github.com/datamllab/tods/blob/benchmark/benchmark/synthetic/Generator.
'''

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def series_segmentation(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def sine(length, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    # timestamp = np.linspace(0, 10, length)
    timestamp = np.arange(length)
    value = np.sin(2 * np.pi * freq * timestamp)
    if noise_amp != 0:
        noise = np.random.normal(0, 1, length)
        value = value + noise_amp * noise
    value = coef * value + offset
    return value


def square_sine(level=5, length=500, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    value = np.zeros(length)
    for i in range(level):
        value += 1 / (2 * i + 1) * sine(length=length, freq=freq * (2 * i + 1), coef=coef, offset=offset, noise_amp=noise_amp)
    return value


def collective_global_synthetic(length, base, coef=1.5, noise_amp=0.005):
    value = []
    norm = np.linalg.norm(base)
    base = base / norm
    num = int(length / len(base))
    for i in range(num):
        value.extend(base)
    residual = length - len(value)
    value.extend(base[:residual])
    value = np.array(value)
    noise = np.random.normal(0, 1, length)
    value = coef * value + noise_amp * noise
    return value


class UnivariateDataGenerator:
    def __init__(self, stream_length, behavior=sine, behavior_config=None):

        self.STREAM_LENGTH = stream_length
        self.behavior = behavior
        self.behavior_config = behavior_config if behavior_config is not None else {}

        self.data = None
        self.label = None
        self.data_origin = None
        self.timestamp = np.arange(self.STREAM_LENGTH)

        self.generate_timeseries()

    def generate_timeseries(self):
        self.behavior_config['length'] = self.STREAM_LENGTH
        self.data = self.behavior(**self.behavior_config)
        self.data_origin = self.data.copy()
        self.label = np.zeros(self.STREAM_LENGTH, dtype=int)

    def point_global_outliers(self, ratio, factor, radius):
        """
        Add point global outliers to original data
        Args:
            ratio: what ratio outliers will be added
            factor: the larger, the outliers are farther from inliers
            radius: the radius of collective outliers range
        """
        position = (np.random.rand(round(self.STREAM_LENGTH * ratio)) * self.STREAM_LENGTH).astype(int)
        maximum, minimum = max(self.data), min(self.data)
        print(maximum, minimum)
        for i in position:
            local_std = self.data_origin[max(0, i - radius):min(i + radius, self.STREAM_LENGTH)].std()
            self.data[i] = self.data_origin[i] * factor * local_std
            if 0 <= self.data[i] < maximum: self.data[i] = maximum
            if 0 > self.data[i] > minimum: self.data[i] = minimum
            self.label[i] = 1

    def point_contextual_outliers(self, ratio, factor, radius):
        """
        Add point contextual outliers to original data
        Args:
            ratio: what ratio outliers will be added
            factor: the larger, the outliers are farther from inliers
                    Notice: point contextual outliers will not exceed the range of [min, max] of original data
            radius: the radius of collective outliers range
        """
        position = (np.random.rand(round(self.STREAM_LENGTH * ratio)) * self.STREAM_LENGTH).astype(int)
        maximum, minimum = max(self.data), min(self.data)
        print(maximum, minimum)
        for i in position:
            local_std = self.data_origin[max(0, i - radius):min(i + radius, self.STREAM_LENGTH)].std()
            self.data[i] = self.data_origin[i] * factor * local_std
            if self.data[i] > maximum: self.data[i] = maximum * min(0.95, abs(np.random.normal(0, 0.5)))  # previous(0, 1)
            if self.data[i] < minimum: self.data[i] = minimum * min(0.95, abs(np.random.normal(0, 0.5)))

            self.label[i] = 1

    def collective_global_outliers(self, ratio, radius, option='square', coef=3., noise_amp=0.0,
                                    level=5, freq=0.04, offset=0.0, # only used when option=='square'
                                    base=[0.,]): # only used when option=='other'
        """
        Add collective global outliers to original data
        Args:
            ratio: what ratio outliers will be added
            radius: the radius of collective outliers range
            option: if 'square': 'level' 'freq' and 'offset' are used to generate square sine wave
                    if 'other': 'base' is used to generate outlier shape
            level: how many sine waves will square_wave synthesis
            base: a list of values that we want to substitute inliers when we generate outliers
        """
        position = (np.random.rand(round(self.STREAM_LENGTH * ratio / (2 * radius))) * self.STREAM_LENGTH).astype(int)

        valid_option = {'square', 'other'}
        if option not in valid_option:
            raise ValueError("'option' must be one of %r." % valid_option)

        if option == 'square':
            sub_data = square_sine(level=level, length=self.STREAM_LENGTH, freq=freq,
                                   coef=coef, offset=offset, noise_amp=noise_amp)
        else:
            sub_data = collective_global_synthetic(length=self.STREAM_LENGTH, base=base,
                                                   coef=coef, noise_amp=noise_amp)
        for i in position:
            start, end = max(0, i - radius), min(self.STREAM_LENGTH, i + radius)
            self.data[start:end] = sub_data[start:end]
            self.label[start:end] = 1

    def collective_trend_outliers(self, ratio, factor, radius):
        """
        Add collective trend outliers to original data
        Args:
            ratio: what ratio outliers will be added
            factor: how dramatic will the trend be
            radius: the radius of collective outliers range
        """
        position = (np.random.rand(round(self.STREAM_LENGTH * ratio / (2 * radius))) * self.STREAM_LENGTH).astype(int)
        for i in position:
            start, end = max(0, i - radius), min(self.STREAM_LENGTH, i + radius)
            slope = np.random.choice([-1, 1]) * factor * np.arange(end - start)
            self.data[start:end] = self.data[start:end] + slope
            self.data[end:] = self.data[end:] + slope[-1]
            self.label[start:end] = 1

    def collective_seasonal_outliers(self, ratio, factor, radius):
        """
        Add collective seasonal outliers to original data
        Args:
            ratio: what ratio outliers will be added
            factor: how many times will frequency multiple
            radius: the radius of collective outliers range
        """
        position = (np.random.rand(round(self.STREAM_LENGTH * ratio / (2 * radius))) * self.STREAM_LENGTH).astype(int)
        seasonal_config = self.behavior_config
        seasonal_config['freq'] = factor * self.behavior_config['freq']
        for i in position:
            start, end = max(0, i - radius), min(self.STREAM_LENGTH, i + radius)
            period_add = factor - 1
            if period_add >= 0:
                new_data = np.tile(self.data[start:end], factor)
                self.data[start:end] = np.array([new_data[i] for i in range(0, len(new_data), factor)])
                self.label[start:end] = 1
            else:
                mul = int(1/factor)
                split = int((end - start) * factor)
                new_data = np.zeros((end - start))
                for i in range(split):
                    new_data[i * mul] =  self.data[start + i]
                
                for i in range((end - start)):
                    if i % mul == 0:
                        continue
                    offset = i % mul
                    if i + mul - offset >= end - start:
                        new_data[i] = new_data[i - offset]
                    else: 
                        new_data[i] = new_data[i - offset] + (new_data[i + mul - offset] - new_data[i - offset]) * offset / mul
                
                self.data[start:end] = new_data
                self.label[start:end] = 1
                
            
        assert len(self.data) == len(self.label)


if __name__ == '__main__':
    np.random.seed(1)

    BEHAVIOR_CONFIG = {'freq': 0.01, 'coef': 1.5, "offset": 0.0, 'noise_amp': 0.02}
    BASE = [1.4529900e-01, 1.2820500e-01, 9.4017000e-02, 7.6923000e-02, 1.1111100e-01, 1.4529900e-01, 1.7948700e-01,
         2.1367500e-01, 2.1367500e-01]

    univariate_data = UnivariateDataGenerator(stream_length=10000, behavior=sine, behavior_config=BEHAVIOR_CONFIG)
    
    ADDTITIONAL_CONFIG = {'path': "datasets/UTS/TODS", 'split_ratio': 0.5, 'ano_ratio': 0.05}
    parser = argparse.ArgumentParser(description='Args for Injection')
    parser.add_argument('--type', type=str, help='Injection type')
    args = parser.parse_args()
    
    curve_name = args.type
    l = len(curve_name)
    if '2' in curve_name:
        univariate_data.collective_global_outliers(ratio=ADDTITIONAL_CONFIG["ano_ratio"]/l, radius=5, option='square', coef=1.5, noise_amp=0.03,
                                                level=20, freq=0.01,
                                                base=BASE, offset=0.0) #2
    if '3' in curve_name:
        # increase frequency
        univariate_data.collective_seasonal_outliers(ratio=ADDTITIONAL_CONFIG["ano_ratio"]/l, factor=2, radius=int(0.5/BEHAVIOR_CONFIG["freq"])) #3
        # decrease freqency
        factor = 0.5
        univariate_data.collective_seasonal_outliers(ratio=ADDTITIONAL_CONFIG["ano_ratio"]/l, factor=factor, radius=int(1/(factor * BEHAVIOR_CONFIG["freq"]))) #3
    if '4' in curve_name:
        univariate_data.collective_trend_outliers(ratio=ADDTITIONAL_CONFIG["ano_ratio"]/l, factor=0.2, radius=5) #4
    if '0' in curve_name:
        univariate_data.point_global_outliers(ratio=ADDTITIONAL_CONFIG["ano_ratio"]/l, factor=1.5, radius=5) #0
    if '1' in curve_name:
        univariate_data.point_contextual_outliers(ratio=ADDTITIONAL_CONFIG["ano_ratio"]/l, factor=2.5, radius=5) #1
        
    split_idx = int(len(univariate_data.data) * ADDTITIONAL_CONFIG["split_ratio"])
    train_data = np.array(univariate_data.data[:split_idx])
    test_data = np.array(univariate_data.data[split_idx:])
    
    train_label = np.array(univariate_data.label[:split_idx])
    test_label = np.array(univariate_data.label[split_idx:])
    
    info_dict = {
        "intervals": 1,
        "training set anomaly ratio": np.count_nonzero(train_label >= 1) / len(train_label),
        "testset anomaly ratio": np.count_nonzero(test_label >= 1) / len(test_label),
        "total anomaly ratio": np.count_nonzero(univariate_data.label >= 1) / len(univariate_data.label)
    }
    
    if not os.path.isdir(ADDTITIONAL_CONFIG["path"]):
        os.mkdir(ADDTITIONAL_CONFIG["path"])
        
    dir_name = os.path.join(ADDTITIONAL_CONFIG["path"], curve_name)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        
    train_path = os.path.join(dir_name, "train.npy")
    train_label_path = os.path.join(dir_name, "train_label.npy")
    test_path = os.path.join(dir_name, "test.npy")
    test_label_path = os.path.join(dir_name, "test_label.npy")
    
    info_path = os.path.join(dir_name, "info.json")
    
    np.save(train_path, train_data)
    np.save(test_path, test_data)
    np.save(train_label_path, train_label)
    np.save(test_label_path, test_label)
    
    with open(info_path, "w") as f:
        json.dump(info_dict, f, indent=4)