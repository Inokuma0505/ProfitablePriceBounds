import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray

def bound_quan(price_list: np.ndarray, q: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    price_list : shape (N, M) の価格データ (N 件のサンプル、M 商品分)
    q          : 上限とする分位数 (例: 0.95 など)

    戻り値:
        lower_bound : (1-q) 分位数 (shape (M,))
        upper_bound : q 分位数   (shape (M,))
    """
    # axis=0 で列ごとに分位数を計算

    # 下限となる分位数を求める
    lower_bound = np.quantile(price_list, 1-q, axis=0)
    # 上限となる分位数を求める
    upper_bound = np.quantile(price_list, q, axis=0)
    bounds_quan = []
    for i in range(len(lower_bound)):
        bounds_quan.append((lower_bound[i], upper_bound[i]))
    return bounds_quan