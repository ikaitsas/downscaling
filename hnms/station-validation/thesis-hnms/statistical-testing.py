# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 11:31:27 2025

@author: Giannis
"""

import numpy as np
import pandas as pd

from scipy.stats import skew
from scipy.stats import shapiro
from scipy.stats import wilcoxon
from scipy.stats import probplot
from scipy.stats import ttest_rel

from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf

from arch.bootstrap import optimal_block_length
from arch.bootstrap import MovingBlockBootstrap
from arch.bootstrap import StationaryBootstrap

import seaborn as sns
import matplotlib.pyplot as plt



