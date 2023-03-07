# -*- coding: utf-8 -*-
import math

import numpy as np
from numpy import log10 as npLog10
from pandas_ta.utils import get_drift, get_offset, verify_series

def variance(arr):
    n = len(arr)
    mean = sum(arr) / n
    var = sum((x - mean) ** 2 for x in arr) / (n - 1)
    return var

def hurst_exponent(close, base_scale=8, max_scale=2, length=18, calculate_sma=False, sma_len=18):
    Close = verify_series(close, length)

    # Create empty array
    fluc = [0] * 10
    scale = [0] * 10

    # function returns the covariance of two arrays
    def covariance(X, Y):
        # Ensure arrays have the same shape
        if X.shape != Y.shape:
            raise ValueError("Input arrays must have the same shape")

        # Compute mean of each array
        X_mean = np.mean(X)
        Y_mean = np.mean(Y)

        # Compute covariance
        cov = np.sum((X - X_mean) * (Y - Y_mean)) / (X.shape[0] - 1)

        return cov

    # Super Smoother Function
    def ss(Series, Period):
        PI = 2.0 * math.asin(1.0)
        SQRT2 = math.sqrt(2.0)
        lmbda = PI * SQRT2 / Period
        a1 = math.exp(-lmbda)
        coeff2 = 2.0 * a1 * math.cos(lmbda)
        coeff3 = - math.pow(a1, 2.0)
        coeff1 = 1.0 - coeff2 - coeff3
        filt1 = 0.0
        filt1 = coeff1 * (Series + Series.shift(1).fillna(method='ffill')) * 0.5 + coeff2 * filt1.shift(1).fillna(
            method='ffill') + coeff3 * filt1.shift(2).fillna(method='ffill')
        return filt1

    # Root Mean Sum (FLuctuation) function linear trend to calculate error between linear trend and cumulative sum
    def RMS(N1, N):
        seq = np.arange(1, N + 1)
        y = Close[N1:N1 + N]

        sdx = np.std(seq) * np.sqrt(N / (N - 1))
        sdy = np.std(y) * np.sqrt(N / (N - 1))
        cov = covariance(seq, y) * (N / (N - 1))

        r2 = (cov / (sdx * sdy)) ** 2
        rms = np.sqrt(1 - r2) * sdy
        return rms

    # Average of Root Mean Sum Measured each block (Log Scale)
    def Arms(bar):
        num = math.floor(length / bar)
        sumr = 0.0
        for i in range(num):
            sumr += RMS(i * bar, bar)
        avg = np.log10(sumr / num)
        return avg

    # Approximating Log Scale Function (Saves Sample Size)
    def fs(x):
        return round(base_scale * math.pow(math.pow(length / (max_scale * base_scale), 0.1111111111), x))

    for i in range(len(fluc)):
        fluc[i] = Arms(fs(i))
    for i in range(len(scale)):
        scale[i] = npLog10(fs(i))

    if calculate_sma == True:
        slope = covariance(scale, fluc) / variance(scale)
        slope.name = f"HURST_EXPONENT_{length}_{base_scale}_{max_scale}"
        slope.category = "trend"
        return ss(slope, sma_len)
    else:
        slope = covariance(scale, fluc) / variance(scale)
        slope.name = f"HURST_EXPONENT_{length}_{base_scale}_{max_scale}"
        slope.category = "trend"
        return slope
