import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray
from scipy.optimize import minimize
from src.price_optimization.price_optimization import predict_objective_function
from src.price_optimization.price_optimization import sales_function
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold




# 目的関数を定義
def cross_validation_bounds_penalty_all(
    bounds: List[float],
    tilda_coefs_list: List[NDArray[np.float_]],
    tilda_intercepts_list: List[NDArray[np.float_]],
    hat_coefs_list: List[NDArray[np.float_]],
    hat_intercepts_list: List[NDArray[np.float_]],
    M: int,
    K: int,
    bounds_range: float,
    lamda_1: float,
    lamda_2:float,
) -> float:
    # bounds の整合性チェック
    bounds_list = []
    for i in range(M):
        # 上下限が逆転していたら平均で固定
        if bounds[i] > bounds[i + M]:
            bounds_mean = (bounds[i] + bounds[i + M]) / 2
            bounds_list.append((bounds_mean, bounds_mean))
        # 上下限が逆転していなければそのまま使用
        else:
            bounds_list.append((bounds[i], bounds[i + M]))
    optimal_sales_list = []

    # すでに外部でKFold分割や学習が終わっているものとして
    # tilda_coefs_list[i], tilda_intercepts_list[i], hat_coefs_list[i], hat_intercepts_list[i]
    # を使用して最適化と売上計算
    for i in range(K):
        intercepts = tilda_intercepts_list[i]
        coefs = tilda_coefs_list[i]

        # 最適化
        initial_prices = np.full(M, 0.6)
        result = minimize(
            predict_objective_function,
            initial_prices,
            args=(intercepts, coefs, M),
            bounds=bounds_list,
            method="L-BFGS-B",
        )
        optimal_prices = result.x

        # hatモデルパラメータで売上計算
        alpha = hat_intercepts_list[i]
        beta = hat_coefs_list[i]

        sales_hat = np.sum(sales_function(optimal_prices, alpha, beta))
        optimal_sales_list.append(sales_hat)

    # ペナルティ計算
    penalty_1 = 0.0
    for i in range(M):
        penalty_1 += bounds[i + M] - bounds[i]

    penalty_2 = 0.0
    for i in range(M):
        penalty_2 += max(0,bounds[i]-bounds[i+M])**2

    mean_sales = np.mean(optimal_sales_list)

    return -mean_sales + lamda_1 * max(0, penalty_1 - M * bounds_range) ** 2 + lamda_2 * penalty_2

# 目的関数を最適化し価格範囲を推定
def estimate_bounds_penalty_nelder_all(
    bounds: List[float],
    tilda_coefs_list: List[NDArray[np.float_]],
    tilda_intercepts_list: List[NDArray[np.float_]],
    hat_coefs_list: List[NDArray[np.float_]],
    hat_intercepts_list: List[NDArray[np.float_]],
    M: int,
    K: int,
    r_min: float,
    r_max: float,
    bounds_range: float,
    lamda_1: float,
    lamda_2: float,
    adaptive: bool = True,
) -> Tuple[float, List[Tuple[float, float]]]:
    # Nelder-Meadでの最適化
    bounds_nelder = minimize(
        cross_validation_bounds_penalty_all,
        bounds,
        args=(
            tilda_coefs_list,
            tilda_intercepts_list,
            hat_coefs_list,
            hat_intercepts_list,
            M,
            K,
            bounds_range,
            lamda_1,
            lamda_2,
        ),
        method="Nelder-Mead",
        bounds=[(r_min, r_max) for _ in range(2 * M)],
        options={"adaptive": adaptive},
    )

    opt_bounds = []
    for i in range(M):
        lb = min(bounds_nelder.x[i], bounds_nelder.x[i + M])
        ub = max(bounds_nelder.x[i], bounds_nelder.x[i + M])
        opt_bounds.append((lb, ub))

    return -bounds_nelder.fun, opt_bounds