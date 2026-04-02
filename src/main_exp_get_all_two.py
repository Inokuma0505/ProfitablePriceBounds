# -*- coding: utf-8 -*-
"""
Auto-generated script from main_exp_get_all.ipynb
"""

import matplotlib.pyplot as plt
import itertools
import time
import numpy as np
from tqdm import tqdm
import json
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from numpy.typing import NDArray
from scipy.optimize import minimize
from typing import List, Tuple
import pandas as pd
from sklearn.metrics import r2_score

# 価格を生成する関数
def create_price(r_mean: float, r_std: float, M: int) -> NDArray[np.float_]:
    # r_mean = (r_min + r_max) / 2
    # r_std = (r_max - r_mean) / 2
    # r_minとr_maxの間のランダムな0.1刻みの少数をM個生成

    # 平均r_meanと標準偏差r_stdを指定して正規分布に従うM個の価格を生成
    price = np.random.normal(r_mean, r_std, size=M)
    # price = np.round(price, 1)

    return price


# alphaを作成する関数
def alpha_star(M: int) -> NDArray[np.float_]:

    # alphaはM個の要素を持つベクトルで、各要素は[-3M, 3M]の範囲で一様分布から生成
    alpha_star = np.random.uniform(M, 3 * M, size=M)
    return alpha_star


# betaを作成する関数
def beta_star(M: int, M_prime: int) -> NDArray[np.float_]:

    # betaはM x M_primeのゼロ行列を作成
    beta_star = np.zeros((M, M_prime))

    for m in range(M):
        for m_prime in range(M_prime):
            # mとm_primeが同じ場合は[-3M, -2M]の範囲で一様分布から生成
            if m == m_prime:
                beta_star[m, m_prime] = np.random.uniform(-3 * M, -2 * M)
            # mとm_primeが異なる場合は[0, 3]の範囲で一様分布から生成
            else:
                beta_star[m, m_prime] = np.random.uniform(0, 3)

    return beta_star


def quantity_function(
    price: NDArray[np.float_],
    alpha: NDArray[np.float_],
    beta: NDArray[np.float_],
    delta: float = 0.1,  # ノイズレベルを指定（例として0.1を使用）
) -> list[float]:
    M = len(price)
    quantity_list = []
    q_m_no_noise = []

    # ステップ1: ノイズなしのq_mを計算
    for m in range(M):
        sum_beta = 0
        for m_prime in range(M):
            sum_beta += beta[m][m_prime] * price[m_prime]
        quantity = alpha[m] + sum_beta
        q_m_no_noise.append(quantity)

    # E[q_m^2]を計算
    E_q_m_squared = np.mean(np.array(q_m_no_noise) ** 2)

    # ステップ2: ノイズの標準偏差sigmaを計算
    sigma = delta * np.sqrt(E_q_m_squared)

    # ステップ3: ノイズを加えて最終的なq_mを計算
    for m in range(M):
        epsilon = np.random.normal(0, sigma)
        quantity = q_m_no_noise[m] + epsilon
        quantity_list.append(quantity)

    return quantity_list


def sales_function(
    price: NDArray[np.float_], alpha: NDArray[np.float_], beta: NDArray[np.float_]
) -> list[float]:
    M = len(price)
    sales_list = []
    # ノイズなしのq_mを計算
    for m in range(M):
        sum_beta = 0
        for m_prime in range(M):
            sum_beta += beta[m][m_prime] * price[m_prime]

        quantity = alpha[m] + sum_beta

        # 需要量に価格をかけて売上を計算
        sales_list.append(quantity * price[m])

    return sales_list


def create_date(M, N, r_mean, r_std, delta=0.1):

    # alphaとbetaを生成
    alpha = alpha_star(M)
    beta = beta_star(M, M)

    # 価格のリストを作成
    price_list = []
    # 需要のリストを作成
    quantity_list = []

    for _ in range(N):

        # 価格を作成
        price = create_price(r_mean, r_std, M)
        # 需要を計算
        quantity = quantity_function(price, alpha, beta, delta)
        # リストに追加
        price_list.append(price)
        quantity_list.append(quantity)
    # 価格と需要をDataFrameに変換
    X = np.array(price_list)
    Y = np.array(quantity_list)

    return alpha, beta, X, Y


def create_bounds(M, r_min, r_max):
    # 価格の下限を設定
    lb = np.full(M, r_min)
    # 価格の上限を設定
    ub = np.full(M, r_max)

    # 提案手法にれる価格範囲
    range_bounds = []
    for i in range(M):
        range_bounds.append(lb[i])

    for i in range(M):
        range_bounds.append(ub[i])
    # 一般的な価格範囲
    bounds = [(r_min, r_max) for _ in range(M)]

    return lb, ub, bounds, range_bounds

# 目的関数を定義（最大化問題を最小化問題に変換）
def sales_objective_function(prices, alpha, beta, M):
    return -sum(
        prices[m] * (alpha[m] + sum(beta[m][m_prime] * prices[m_prime] for m_prime in range(M)))
        for m in range(M)
    )


def sales_optimize(
    M: int,
    alpha: np.ndarray,
    beta: np.ndarray,
    bounds: list[tuple[float, float]],
) -> Tuple[float, np.ndarray]:
    # 初期値として与えられたprices_listを使用
    initial_prices = np.full(M, 0.6)

    # 最適化を実行
    result = minimize(
        sales_objective_function,
        initial_prices,
        args=(alpha, beta, M),
        bounds=bounds,
        method="L-BFGS-B",
    )
    # 最適な価格と目的関数の値を取得
    optimal_prices = result.x
    optimal_value = -result.fun  # 符号を反転して元の最大化問題の値に戻す
    return optimal_value, optimal_prices

# 目的関数を定義
def predict_objective_function(
    prices: NDArray[np.float_], intercepts: [float], coefs: [NDArray[np.float_]], M: int
) -> float:
    # 各変数の内容をデバッグ出力
    # print("prices:", prices)
    # print("intercepts:", intercepts)
    # print("coefs:", coefs)
    # print("M:", M)

    return -sum(
        prices[m]
        * (intercepts[m] + sum(coefs[m][m_prime] * prices[m_prime] for m_prime in range(M)))
        for m in range(M)
    )


# 予測と最適化を行う関数
def predict_optimize(
    M: int, X: NDArray[np.float_], Y: NDArray[np.float_], bounds: list[float]
) -> tuple[float, NDArray[np.float_]]:
    lr = MultiOutputRegressor(LinearRegression())
    lr.fit(X, Y)
    # 係数と切片を取得
    coefs = [estimate.coef_ for estimate in lr.estimators_]
    intercepts = [estimate.intercept_ for estimate in lr.estimators_]

    # 初期値として与えられたprices_listを使用
    initial_prices = np.full(M, 0.6)
    # 最適化を実行
    result = minimize(
        predict_objective_function,
        initial_prices,
        args=(intercepts, coefs, M),
        bounds=bounds,
        method="L-BFGS-B",
    )
    # 最適な価格と目的関数の値を取得
    optimal_prices = result.x
    optimal_value = -result.fun  # 符号を反転して元の最大化問題の値に戻す
    return optimal_value, optimal_prices

# CVを行う関数
def cross_validation(
    tilda_coefs_list: list[NDArray[np.float_]],
    tilda_intercepts_list: list[float],
    hat_coefs_list: list[NDArray[np.float_]],
    hat_intercepts_list: list[float],
    M: int,
    K: int,
    bounds: list[float],
) -> float:
    optimal_sales_list = []
    optimal_prices_list = [[] for _ in range(M)]
    for i in range(K):
        # 初期値として与えられたprices_listを使用
        initial_prices = np.full(M, 0.6)

        # 最適化を実行
        result = minimize(
            predict_objective_function,
            initial_prices,
            args=(tilda_intercepts_list[i], tilda_coefs_list[i], M),
            bounds=bounds,
            method="L-BFGS-B",
        )
        # 最適な価格と目的関数の値を取得
        optimal_prices = result.x
        # print("optimal_prices cv:", optimal_prices)
        for m in range(M):
            optimal_prices_list[m].append(optimal_prices[m])

        sales_hat = np.sum(
            sales_function(optimal_prices, hat_intercepts_list[i], hat_coefs_list[i])
        )

        optimal_sales_list.append(sales_hat)

    return np.mean(optimal_sales_list), optimal_prices_list

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

def bootstrap_bounds(
    M: int,
    X: np.ndarray,
    Y: np.ndarray,
    r_min: float,
    r_max: float,
    n_iterations: int = 1000,
    k: float = 1.96
) -> tuple[np.ndarray, np.ndarray]:
    """
    ブートストラップサンプルを用いて各商品の最適価格の統計量（平均±k*標準偏差）から価格範囲を算出する関数
    
    Parameters:
      M: 商品数（価格の次元数）
      X: 価格設定のデータ（各行が一つの実験データ、shape=(n_samples, M)）
      Y: 需要のデータ（Xと対応するデータ、shape=(n_samples, M)）
      bounds: 最適化に使用する各商品の価格下限・上限のリスト（例：[(r_min, r_max), ...]）
      n_iterations: ブートストラップの反復回数（デフォルトは1000）
      k: 標準偏差のスケールパラメータ（例：1.96 なら約95%信頼区間）
    
    Returns:
      lower_bounds: 各商品の価格下限（mean - k * std）
      upper_bounds: 各商品の価格上限（mean + k * std）
      
    ※ 内部で predict_optimize 関数を使用して最適価格を算出している前提です。
    """
    bounds = [(r_min, r_max) for _ in range(M)]
    optimal_prices_list = []
    n_samples = X.shape[0]
    
    for i in range(n_iterations):
        # ブートストラップサンプルを行単位の復元抽出で取得
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_bs = X[indices]
        Y_bs = Y[indices]
        
        # 取得したブートストラップサンプルを用いて価格最適化を実施
        # predict_optimize は (optimal_value, optimal_prices) を返す前提
        _, opt_prices = predict_optimize(M, X_bs, Y_bs, bounds)
        optimal_prices_list.append(opt_prices)
    
    # ブートストラップで得られた最適価格を NumPy 配列に変換（shape: (n_iterations, M)）
    #print(optimal_prices_list)
    optimal_prices_array = np.array(optimal_prices_list)
    
    # 各商品の最適価格の平均と標準偏差を計算
    mean_prices = np.mean(optimal_prices_array, axis=0)
    std_prices = np.std(optimal_prices_array, axis=0)
    
    # 平均 ± k * 標準偏差を下限・上限として算出
    lower_bounds = mean_prices - k * std_prices
    upper_bounds = mean_prices + k * std_prices
    
    # 結果をタプルに格納
    bootstrap_bounds = []
    for i in range(M):
        
        # r_min と r_max でクリッピング
        lower = max(r_min, lower_bounds[i])
        upper = min(r_max, upper_bounds[i])

        bootstrap_bounds.append((lower, upper))

    return bootstrap_bounds

# 実験設定
config = {
    "M_list": [5, 10],
    "delta_list": [0.6, 0.8, 1.0],
    "N": 500,
    "K": 5,
    "B": 100,
    "r_mean": 0.8,
    "r_std": 0.1,
    "r_min": 0.5,
    "r_max": 1.1
}

# 各手法のパラメータ定義
quantiles = [0.95, 0.90, 0.85, 0.80]
boot_k = {99: 2.576, 95: 1.96, 90: 1.645}
penalties = {'ebpa3': 0.30, 'ebpa4': 0.40, 'ebpa5': 0.50,'ebpa6': 0.60}

# 結果格納用辞書（M, delta の組み合わせごと）
results = {}
param_combos = list(itertools.product(config["M_list"], config["delta_list"]))
for M, delta in param_combos:
    key = f"M{M}_delta{delta}"
    results[key] = {}
    methods = ['so', 'po']
    methods += [f'quan{int(q*100)}' for q in quantiles]
    methods += [f'boot{p}' for p in boot_k]
    methods += list(penalties.keys())
    for m in methods:
        results[key][m] = {
            'sales_ratio': [],
            'true_sales_ratio': [],
            'range_diff': [],
            'range_per_product_diff': [],
            'range_per_product': [],
            'prices': [],
            'time': []
        }
    results[key]['r2_list'] = []

# 総実験数および進捗バー
total_runs = len(param_combos) * 100
bar = tqdm(total=total_runs, desc="Total Experiments")

# メイン実験ループ
for M, delta in param_combos:
    key = f"M{M}_delta{delta}"
    lb, ub, bounds, range_bounds = create_bounds(M, config["r_min"], config["r_max"])

    for i in range(100):
        # 乱数シード設定
        seed_offset = int(delta * 10) + (10 if M == 10 else 0)
        np.random.seed(i + seed_offset)
        alpha, beta, X, Y = create_date(M, config["N"], config["r_mean"], config["r_std"], delta)

        # 交差検証用に切片と係数を格納するlistを作成
        tilda_coefs_list = []
        tilda_intercepts_list = []
        hat_coefs_list = []
        hat_intercepts_list = []

        kf = KFold(n_splits=config["K"], shuffle=True, random_state=0)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            # lr_tilda: trainデータで学習
            lr_tilda = MultiOutputRegressor(LinearRegression())
            lr_tilda.fit(X_train, y_train)

            # tildaの係数・切片を保存
            coefs = [est.coef_ for est in lr_tilda.estimators_]
            intercepts = [est.intercept_ for est in lr_tilda.estimators_]
            tilda_coefs_list.append(coefs)
            tilda_intercepts_list.append(intercepts)

            # lr_hat: testデータで学習
            lr_hat = MultiOutputRegressor(LinearRegression())
            lr_hat.fit(X_test, y_test)

            hat_coefs = [est.coef_ for est in lr_hat.estimators_]
            hat_intercepts = [est.intercept_ for est in lr_hat.estimators_]
            hat_coefs_list.append(hat_coefs)
            hat_intercepts_list.append(hat_intercepts)

        # SO
        start = time.perf_counter()
        so_sales, so_prices = sales_optimize(M, alpha, beta, bounds)
        elapsed = time.perf_counter() - start
        per_prod_range = [b[1] - b[0] for b in bounds]
        results[key]['so']['time'].append(elapsed)
        results[key]['so']['sales_ratio'].append(so_sales / so_sales)
        results[key]['so']['true_sales_ratio'].append(
            np.sum(sales_function(so_prices, alpha, beta)) / so_sales
        )
        results[key]['so']['range_diff'].append(sum(per_prod_range))
        results[key]['so']['range_per_product_diff'].append(per_prod_range)
        results[key]['so']['range_per_product'].append(bounds)
        results[key]['so']['prices'].append(so_prices.tolist())

        # PO
        start = time.perf_counter()
        po_sales, po_prices = predict_optimize(M, X, Y, bounds)
        elapsed = time.perf_counter() - start
        results[key]['po']['time'].append(elapsed)
        results[key]['po']['sales_ratio'].append(po_sales / so_sales)
        results[key]['po']['true_sales_ratio'].append(
            np.sum(sales_function(po_prices, alpha, beta)) / so_sales
        )
        results[key]['po']['range_diff'].append(sum(per_prod_range))
        results[key]['po']['range_per_product_diff'].append(per_prod_range)
        results[key]['po']['range_per_product'].append(bounds)
        results[key]['po']['prices'].append(po_prices.tolist())

        # 分位数法
        for q in quantiles:
            subkey = f'quan{int(q*100)}'
            start = time.perf_counter()
            bounds_q = bound_quan(X, q)
            sales_q, prices_q = predict_optimize(M, X, Y, bounds_q)
            elapsed = time.perf_counter() - start
            true_q = np.sum(sales_function(prices_q, alpha, beta))
            per_prod_range_q = [b[1] - b[0] for b in bounds_q]
            results[key][subkey]['time'].append(elapsed)
            results[key][subkey]['sales_ratio'].append(sales_q / so_sales)
            results[key][subkey]['true_sales_ratio'].append(true_q / so_sales)
            results[key][subkey]['range_diff'].append(sum(per_prod_range_q))
            results[key][subkey]['range_per_product_diff'].append(per_prod_range_q)
            results[key][subkey]['range_per_product'].append(bounds_q)
            results[key][subkey]['prices'].append(prices_q.tolist())

        # ブートストラップ法
        for p_val, k in boot_k.items():
            subkey = f'boot{p_val}'
            start = time.perf_counter()
            bounds_b = bootstrap_bounds(M, X, Y, config["r_min"], config["r_max"], n_iterations=config["B"], k=k)
            sales_b, prices_b = predict_optimize(M, X, Y, bounds_b)
            elapsed = time.perf_counter() - start
            true_b = np.sum(sales_function(prices_b, alpha, beta))
            per_prod_range_b = [b[1] - b[0] for b in bounds_b]
            results[key][subkey]['time'].append(elapsed)
            results[key][subkey]['sales_ratio'].append(sales_b / so_sales)
            results[key][subkey]['true_sales_ratio'].append(true_b / so_sales)
            results[key][subkey]['range_diff'].append(sum(per_prod_range_b))
            results[key][subkey]['range_per_product_diff'].append(per_prod_range_b)
            results[key][subkey]['range_per_product'].append(bounds_b)
            results[key][subkey]['prices'].append(prices_b.tolist())

        # ペナルティ法
        for pen_key, pen in penalties.items():
            start = time.perf_counter()
            _, bounds_p = estimate_bounds_penalty_nelder_all(
                range_bounds,
                tilda_coefs_list, tilda_intercepts_list, hat_coefs_list, hat_intercepts_list,
                M, config["K"], config["r_min"], config["r_max"], pen, 1.0, 1.0
            )
            sales_p, prices_p = predict_optimize(M, X, Y, bounds_p)
            elapsed = time.perf_counter() - start
            true_p = np.sum(sales_function(prices_p, alpha, beta))
            per_prod_range_p = [b[1] - b[0] for b in bounds_p]
            results[key][pen_key]['time'].append(elapsed)
            results[key][pen_key]['sales_ratio'].append(sales_p / so_sales)
            results[key][pen_key]['true_sales_ratio'].append(true_p / so_sales)
            results[key][pen_key]['range_diff'].append(sum(per_prod_range_p))
            results[key][pen_key]['range_per_product_diff'].append(per_prod_range_p)
            results[key][pen_key]['range_per_product'].append(bounds_p)
            results[key][pen_key]['prices'].append(prices_p.tolist())

        # 全体モデルR2
        lr = MultiOutputRegressor(LinearRegression()).fit(X, Y)
        preds = lr.predict(X)
        for idx in range(M):
            results[key]['r2_list'].append(r2_score(Y[:, idx], preds[:, idx]))

        # 進捗更新
        bar.update(1)

bar.close()
# 終了後: results に全データが格納されています

# JSON 形式でファイル出力
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)





