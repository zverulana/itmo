"""
Модуль с импортами для VAE + KMeans + Boundary Clustering
Используйте: from imports import *
или импортируйте конкретные модули по необходимости
"""

# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import warnings
import time
from collections import Counter

# PyTorch для VAE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Scikit-learn для кластеризации и метрик
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.neighbors import NearestNeighbors

# SciPy для расстояний, энтропии и статистических тестов
from scipy.spatial.distance import cdist
from scipy.stats import entropy, wilcoxon, mannwhitneyu, friedmanchisquare, rankdata

# Для коррекции на множественные сравнения
try:
    from statsmodels.stats.multitest import multipletests
    MULTIPLE_TESTING_AVAILABLE = True
except ImportError:
    MULTIPLE_TESTING_AVAILABLE = False

# Bayesian Optimization
try:
    from skopt import gp_minimize
    from skopt.space import Integer, Real
    from skopt.utils import use_named_args
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False

# Fuzzy кластеризация (опционально)
try:
    import skfuzzy as fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    # Не выводим предупреждение, так как это опциональная зависимость

warnings.filterwarnings('ignore')

# Настройка визуализации
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 12

# Глобальные константы
RANDOM_STATE = 22
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

