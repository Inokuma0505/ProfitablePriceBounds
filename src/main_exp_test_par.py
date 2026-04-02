# -*- coding: utf-8 -*-
"""
Optimized experiment script for main_exp_get_all_two.py
  - 元の価格生成を保持
  - NumPy ベクトル化
  - Joblib 並列化 + tqdmプログレスバー
  - ペナルティ法追加
"""
import itertools
import time
import json
import argparse
import os
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from typing import Tuple

# --- 価格, alpha, beta 生成関数（元の実装） ---
def create_price(r_mean: float, r_std: float, M: int) -> np.ndarray:
    return np.random.normal(r_mean, r_std, size=M)

def alpha_star(M: int) -> np.ndarray:
    return np.random.uniform(M, 3 * M, size=M)

def beta_star(M: int) -> np.ndarray:
    beta = np.zeros((M, M))
    for m in range(M):
        for m_prime in range(M):
            if m == m_prime:
                beta[m, m_prime] = np.random.uniform(-3 * M, -2 * M)
            else:
                beta[m, m_prime] = np.random.uniform(0, 3)
    return beta

# --- 需要量, 売上関数 ---
def quantity_function(
    price: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    delta: float = 0.1
) -> np.ndarray:
    q_no = alpha + beta.dot(price)
    sigma = delta * np.sqrt(np.mean(q_no ** 2))
    noise = np.random.normal(0, sigma, size=q_no.shape)
    return q_no + noise

def sales_function(price: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return price * (alpha + beta.dot(price))

# --- データ生成 ---
def create_date(M: int, N: int, r_mean: float, r_std: float, delta: float = 0.1):
    alpha = alpha_star(M)
    beta = beta_star(M)
    prices, quantities = [], []
    for _ in range(N):
        p = create_price(r_mean, r_std, M)
        q = quantity_function(p, alpha, beta, delta)
        prices.append(p)
        quantities.append(q)
    return alpha, beta, np.array(prices), np.array(quantities)

# --- 境界設定 ---
def create_bounds(M: int, r_min: float, r_max: float):
    lb = np.full(M, r_min)
    ub = np.full(M, r_max)
    return lb, ub, [(r_min, r_max)] * M, np.concatenate([lb, ub])


DEFAULT_FIXED_WIDTHS = [
    0.60, 0.55, 0.50, 0.45, 0.40, 0.35,
    0.30, 0.25, 0.20, 0.15, 0.10, 0.05,
]


def fixed_width_name(width: float) -> str:
    return f"fixed{int(round(width * 100)):02d}"


def elastic_opt_name(width: float) -> str:
    return f"elaopt{int(round(width * 100)):02d}"


def elastic_bias_name(width: float) -> str:
    return f"elabias{int(round(width * 100)):02d}"


def create_fixed_bounds(
    M: int,
    center: float,
    width: float,
    r_min: float,
    r_max: float,
) -> list[tuple[float, float]]:
    half_width = width / 2
    lower = max(r_min, center - half_width)
    upper = min(r_max, center + half_width)
    return [(lower, upper)] * M


def create_centered_bounds_from_centers(
    centers: np.ndarray,
    width: float,
    r_min: float,
    r_max: float,
) -> list[tuple[float, float]]:
    bounds = []
    for center in centers:
        lower = center - width * (center - r_min)
        upper = center + width * (r_max - center)
        bounds.append((lower, upper))
    return bounds


def create_elasticity_biased_bounds(
    elasticities: np.ndarray,
    width: float,
    r_min: float,
    r_max: float,
) -> list[tuple[float, float]]:
    mean_elasticity = float(np.mean(elasticities))
    bounds = []
    for elasticity in elasticities:
        if elasticity <= mean_elasticity:
            lower = max(r_min, r_max - width)
            upper = r_max
        else:
            lower = r_min
            upper = min(r_max, r_min + width)
        bounds.append((lower, upper))
    return bounds


def optimize_fitted_model(
    M: int,
    bounds,
    coefs,
    intercepts,
    init: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    if init is None:
        init = np.full(M, 0.6)
    res = minimize(
        lambda p: -np.sum(p * (np.array(intercepts) + np.stack(coefs).dot(p))),
        init,
        bounds=bounds,
        method="L-BFGS-B",
    )
    return -res.fun, res.x


def optimize_single_product_revenue(
    intercept: float,
    own_coef: float,
    r_min: float,
    r_max: float,
) -> float:
    candidates = [r_min, r_max]
    if abs(own_coef) > 1e-12:
        stationary = float(np.clip(-intercept / (2 * own_coef), r_min, r_max))
        candidates.append(stationary)
    revenues = [price * (intercept + own_coef * price) for price in candidates]
    return float(candidates[int(np.argmax(revenues))])


def estimate_independent_optimal_centers(
    coefs,
    intercepts,
    r_min: float,
    r_max: float,
) -> np.ndarray:
    centers = []
    for idx, intercept in enumerate(intercepts):
        centers.append(
            optimize_single_product_revenue(
                float(intercept),
                float(coefs[idx][idx]),
                r_min,
                r_max,
            )
        )
    return np.array(centers)


def estimate_own_price_elasticities(
    coefs,
    intercepts,
    reference_price: float,
) -> np.ndarray:
    elasticities = []
    for idx, intercept in enumerate(intercepts):
        own_coef = float(coefs[idx][idx])
        predicted_quantity = float(intercept) + own_coef * reference_price
        denom = max(abs(predicted_quantity), 1e-8)
        elasticities.append(abs(own_coef * reference_price / denom))
    return np.array(elasticities)


def parse_fixed_widths(raw_widths: str | None) -> list[float]:
    if raw_widths is None:
        return DEFAULT_FIXED_WIDTHS.copy()

    widths = []
    for token in raw_widths.split(","):
        token = token.strip()
        if not token:
            continue
        width = float(token)
        if width <= 0:
            raise ValueError(f"fixed width must be positive: {token}")
        widths.append(width)

    if not widths:
        raise ValueError("at least one fixed width must be specified")

    return widths


def build_additional_method_names(
    fixed_widths,
    run_fixed: bool,
    run_elastic: bool,
) -> list[str]:
    methods = []
    if run_fixed:
        methods += [fixed_width_name(width) for width in fixed_widths]
    if run_elastic:
        methods += [elastic_opt_name(width) for width in fixed_widths]
        methods += [elastic_bias_name(width) for width in fixed_widths]
    return methods


def build_method_names(
    quantiles,
    boot_k,
    penalties,
    fixed_widths,
    run_existing: bool,
    run_fixed: bool,
    run_elastic: bool,
) -> list[str]:
    methods = ['so', 'po']
    methods += build_additional_method_names(fixed_widths, run_fixed, run_elastic)
    if not run_existing:
        return methods

    methods += [f'quan{int(q*100)}' for q in quantiles]
    methods += [f'boot{p}' for p in boot_k]
    methods += list(penalties.keys())
    return methods


def merge_results(existing_results, new_results, method_names) -> dict:
    for key, methods in new_results.items():
        if key not in existing_results:
            existing_results[key] = {}
        for method_name in method_names:
            if method_name not in methods:
                continue
            existing_results[key][method_name] = methods[method_name]
    return existing_results


def infer_paper_output_path(config) -> str:
    m_list = config['M_list']
    if len(m_list) != 1:
        raise ValueError("automatic paper filename requires exactly one value in M_list")

    m_value = m_list[0]
    n_value = config['N']
    if m_value == 5:
        return f"paper_{n_value}.json"
    return f"paper_{m_value}_{n_value}.json"


def resolve_output_path(default_output: str, only_additional_mode: bool, output: str | None) -> str:
    if output:
        return output
    if not only_additional_mode:
        return default_output

    stem, ext = os.path.splitext(default_output)
    return f"{stem}_fixed_only{ext}"


def parse_args(default_output: str):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only-fixed",
        action="store_true",
        help="Run only SO, PO, and the fixed-width methods.",
    )
    parser.add_argument(
        "--only-elastic",
        action="store_true",
        help="Run only SO, PO, and the elasticity-based methods.",
    )
    parser.add_argument(
        "--fixed-widths",
        default=None,
        help="Comma-separated total widths for fixed-width and elasticity-based bounds. Defaults to 0.60,0.55,...,0.05.",
    )
    parser.add_argument(
        "--fixed-center",
        type=float,
        default=0.8,
        help="Center price used by the fixed-width bounds.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Output JSON path. Default: {default_output} (or *_fixed_only.json with --only-fixed).",
    )
    parser.add_argument(
        "--merge-into",
        default=None,
        help="Merge computed methods into an existing JSON file. If --output is omitted, the file is updated in place.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Override the number of generated samples per experiment.",
    )
    return parser.parse_args()

# --- モデル予測＋最適化 ---
def predict_optimize(M: int, X: np.ndarray, Y: np.ndarray, bounds):
    lr = MultiOutputRegressor(LinearRegression()).fit(X, Y)
    coefs = [est.coef_ for est in lr.estimators_]
    inter = [est.intercept_ for est in lr.estimators_]
    init = np.full(M, 0.6)
    res = minimize(
        lambda p: -np.sum(p * (np.array(inter) + np.stack(coefs).dot(p))),
        init, bounds=bounds, method="L-BFGS-B"
    )
    return -res.fun, res.x

# --- ペナルティ法関連 ---
def cross_validation_bounds_penalty_all(
    bounds: np.ndarray,
    tilda_coefs_list, tilda_intercepts_list,
    hat_coefs_list, hat_intercepts_list,
    M: int, K: int,
    bounds_range: float, lam1: float, lam2: float
):
    
    bounds_list = []
    for i in range(M):
        # 上下限が逆転していたら平均で固定
        if bounds[i] > bounds[i + M]:
            bounds_mean = (bounds[i] + bounds[i + M]) / 2
            bounds_list.append((bounds_mean, bounds_mean))
        # 上下限が逆転していなければそのまま使用
        else:
            bounds_list.append((bounds[i], bounds[i + M]))    
    sales_vals = []
    for j in range(K):
        init = np.full(M, 0.6)
        res = minimize(
            lambda p: -np.sum(p * (np.array(tilda_intercepts_list[j]) + np.stack(tilda_coefs_list[j]).dot(p))),
            init, bounds=bounds_list, method="L-BFGS-B"
        )
        p_opt = res.x
        sales_vals.append(np.sum(
            sales_function(
                p_opt,
                np.array(hat_intercepts_list[j]),
                np.stack(hat_coefs_list[j])
            )
        ))
    mean_sales = np.mean(sales_vals)
    pen1 = 0
    for i in range(M):
        pen1 += bounds[i + M] - bounds[i]
    pen2 = 0
    for i in range(M):
        pen2 += max(0,bounds[i]-bounds[i+M])**2
    return -mean_sales + lam1 * max(0, pen1 - M * bounds_range) ** 2 + lam2 * pen2

def estimate_bounds_penalty_nelder_all(
    range_bounds: np.ndarray,
    tilda_coefs_list, tilda_intercepts_list,
    hat_coefs_list, hat_intercepts_list,
    M: int, K: int,
    r_min: float, r_max: float,
    bounds_range: float, lam1: float, lam2: float,
    adaptive: bool = True
):
    res = minimize(
        cross_validation_bounds_penalty_all,
        range_bounds,
        args=(tilda_coefs_list, tilda_intercepts_list, hat_coefs_list, hat_intercepts_list, M, K, bounds_range, lam1, lam2),
        method="Nelder-Mead",
        bounds=[(r_min, r_max)] * (2 * M),
        options={"adaptive": adaptive}
    )
    opt = res.x
    return -res.fun, [(min(opt[i], opt[i+M]), max(opt[i], opt[i+M])) for i in range(M)]

# --- 単一実行 ---
from typing import Tuple, Union, List
import numpy as np

def bound_quan(
    price_list: np.ndarray,
    q: float,
    r_min: Union[float, np.ndarray],
    r_max: Union[float, np.ndarray]
) -> List[Tuple[float, float]]:
    """
    price_list : shape (N, M) の価格データ (N 件のサンプル、M 商品分)
    q          : 上限とする分位数 (例: 0.95 など)
    r_min      : クリッピングする下限値 (スカラー or shape (M,))
    r_max      : クリッピングする上限値 (スカラー or shape (M,))

    戻り値:
        bounds_quan : 各商品ごとに (lower, upper) のタプルを格納したリスト
    """
    # axis=0 で列ごとに分位数を計算
    lower_bound = np.quantile(price_list, 1 - q, axis=0)
    upper_bound = np.quantile(price_list, q,     axis=0)

    # 下限・上限をそれぞれ [r_min, r_max] の範囲内にクリッピング
    lower_clipped = np.clip(lower_bound, r_min, r_max)
    upper_clipped = np.clip(upper_bound, r_min, r_max)

    # 結果をリストのタプル形式にまとめる
    bounds_quan: List[Tuple[float, float]] = [
        (lower_clipped[i], upper_clipped[i]) for i in range(lower_clipped.shape[0])
    ]
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

# --- 単一実行 ---
def run_one(
    i,
    M,
    delta,
    config,
    quantiles,
    boot_k,
    penalties,
    fixed_widths,
    fixed_center,
    only_fixed,
    only_elastic,
):
    run_existing = not (only_fixed or only_elastic)
    run_fixed = (not only_elastic) or only_fixed
    run_elastic = (not only_fixed) or only_elastic
    if M == 5 :
        seed_int = int(delta*10) + i
    else:
        seed_int = int(delta*10) + M + i
    np.random.seed(seed_int)
    alpha, beta, X, Y = create_date(M, config['N'], config['r_mean'], config['r_std'], delta)
    _, _, bounds, range_bounds = create_bounds(M, config['r_min'], config['r_max'])

    out = {}
    # SO
    t0 = time.perf_counter()
    r = minimize(lambda p: -np.sum(p*(alpha+beta.dot(p))), np.full(M,0.6), bounds=bounds, method='L-BFGS-B')
    so_val, so_p = -r.fun, r.x
    t1 = time.perf_counter()
    out['so'] = {
        'time': t1-t0,
        'sales_ratio': 1.0,
        'true_sales_ratio': np.sum(sales_function(so_p, alpha, beta)) / so_val,
        'range_diff': M*(config['r_max']-config['r_min']),
        'range_per_product_diff': [config['r_max']-config['r_min']]*M,
        'range_per_product': bounds,
        'prices': so_p.tolist()
    }
    # --- フルデータで再学習したhatモデルの係数・切片を取得 ---
    lr_full = MultiOutputRegressor(LinearRegression()).fit(X, Y)
    coefs_full = [est.coef_ for est in lr_full.estimators_]
    ints_full  = [est.intercept_ for est in lr_full.estimators_]
    init = np.full(M, 0.6)

    # PO
    t0 = time.perf_counter()
    po_val, po_p = optimize_fitted_model(M, bounds, coefs_full, ints_full, init)
    t1 = time.perf_counter()
    out['po'] = {
        'time': t1-t0,
        'sales_ratio': po_val/so_val,
        'true_sales_ratio': np.sum(sales_function(po_p, alpha, beta)) / so_val,
        'range_diff': M*(config['r_max']-config['r_min']),
        'range_per_product_diff': [config['r_max']-config['r_min']]*M,
        'range_per_product': bounds,
        'prices': po_p.tolist()
    }

    if run_elastic:
        independent_centers = estimate_independent_optimal_centers(
            coefs_full,
            ints_full,
            config['r_min'],
            config['r_max'],
        )
        own_elasticities = estimate_own_price_elasticities(
            coefs_full,
            ints_full,
            fixed_center,
        )

    # --- 固定幅法 (fixed-width) ---
    if run_fixed:
        for width in fixed_widths:
            t0_f = time.perf_counter()
            bounds_f = create_fixed_bounds(
                M,
                fixed_center,
                width,
                config['r_min'],
                config['r_max'],
            )
            lower_f = np.array([l for l, _ in bounds_f])
            upper_f = np.array([u for _, u in bounds_f])
            fixed_val, fixed_p = optimize_fitted_model(M, bounds_f, coefs_full, ints_full, init)
            t1_f = time.perf_counter()
            out[fixed_width_name(width)] = {
                'time': t1_f - t0_f,
                'sales_ratio': fixed_val / so_val,
                'true_sales_ratio': np.sum(sales_function(fixed_p, alpha, beta)) / so_val,
                'range_diff': np.sum(upper_f - lower_f),
                'range_per_product_diff': (upper_f - lower_f).tolist(),
                'range_per_product': bounds_f,
                'prices': fixed_p.tolist()
            }

    if run_elastic:
        for width in fixed_widths:
            t0_e1 = time.perf_counter()
            bounds_e1 = create_centered_bounds_from_centers(
                independent_centers,
                width,
                config['r_min'],
                config['r_max'],
            )
            lower_e1 = np.array([l for l, _ in bounds_e1])
            upper_e1 = np.array([u for _, u in bounds_e1])
            elaopt_val, elaopt_p = optimize_fitted_model(M, bounds_e1, coefs_full, ints_full, init)
            t1_e1 = time.perf_counter()
            out[elastic_opt_name(width)] = {
                'time': t1_e1 - t0_e1,
                'sales_ratio': elaopt_val / so_val,
                'true_sales_ratio': np.sum(sales_function(elaopt_p, alpha, beta)) / so_val,
                'range_diff': np.sum(upper_e1 - lower_e1),
                'range_per_product_diff': (upper_e1 - lower_e1).tolist(),
                'range_per_product': bounds_e1,
                'prices': elaopt_p.tolist()
            }

            t0_e2 = time.perf_counter()
            bounds_e2 = create_elasticity_biased_bounds(
                own_elasticities,
                width,
                config['r_min'],
                config['r_max'],
            )
            lower_e2 = np.array([l for l, _ in bounds_e2])
            upper_e2 = np.array([u for _, u in bounds_e2])
            elabias_val, elabias_p = optimize_fitted_model(M, bounds_e2, coefs_full, ints_full, init)
            t1_e2 = time.perf_counter()
            out[elastic_bias_name(width)] = {
                'time': t1_e2 - t0_e2,
                'sales_ratio': elabias_val / so_val,
                'true_sales_ratio': np.sum(sales_function(elabias_p, alpha, beta)) / so_val,
                'range_diff': np.sum(upper_e2 - lower_e2),
                'range_per_product_diff': (upper_e2 - lower_e2).tolist(),
                'range_per_product': bounds_e2,
                'prices': elabias_p.tolist()
            }

    if run_existing:
        # KFold の学習を並列化
        kf = KFold(n_splits=config['K'], shuffle=True, random_state=0)
        splits = list(kf.split(X))

        def process_fold(split):
            tr, te = split
            # tilda モデル
            lr_t = MultiOutputRegressor(LinearRegression()).fit(X[tr], Y[tr])
            t_cc = [e.coef_ for e in lr_t.estimators_]
            t_ii = [e.intercept_ for e in lr_t.estimators_]
            # hat モデル
            lr_h = MultiOutputRegressor(LinearRegression()).fit(X[te], Y[te])
            h_cc = [e.coef_ for e in lr_h.estimators_]
            h_ii = [e.intercept_ for e in lr_h.estimators_]
            return t_cc, t_ii, h_cc, h_ii

        results_fold = Parallel(n_jobs=config['K'])(delayed(process_fold)(split) for split in splits)
        t_coefs, t_ints, h_coefs, h_ints = zip(*results_fold)
        t_coefs, t_ints, h_coefs, h_ints = list(t_coefs), list(t_ints), list(h_coefs), list(h_ints)

        # --- 分位点法 (quantiles) ---
        for q in quantiles:
            t0_q = time.perf_counter()
            bounds_q = bound_quan(X, q, r_min=config['r_min'], r_max=config['r_max'])
            lower = np.array([l for l, u in bounds_q])
            upper = np.array([u for l, u in bounds_q])
            q_val, q_p = predict_optimize(M, X, Y, bounds=bounds_q)
            t1_q = time.perf_counter()
            out[f'quan{int(q*100)}'] = {
                'time': t1_q - t0_q,
                'sales_ratio': q_val / so_val,
                'true_sales_ratio': np.sum(sales_function(q_p, alpha, beta)) / so_val,
                'range_diff': np.sum(upper - lower),
                'range_per_product_diff': (upper - lower).tolist(),
                'range_per_product': bounds_q,
                'prices': q_p.tolist()
            }

        # --- ブートストラップ法 (bootstrap) ---
        for k, z in boot_k.items():
            t0_b = time.perf_counter()
            bounds_b = bootstrap_bounds(
                M,
                X,
                Y,
                config['r_min'],
                config['r_max'],
                n_iterations=config['B'],
                k=z
            )
            lower_b = np.array([l for l, u in bounds_b])
            upper_b = np.array([u for l, u in bounds_b])
            b_val, b_p = predict_optimize(M, X, Y, bounds=bounds_b)
            t1_b = time.perf_counter()
            out[f'boot{k}'] = {
                'time': t1_b - t0_b,
                'sales_ratio': b_val / so_val,
                'true_sales_ratio': np.sum(sales_function(b_p, alpha, beta)) / so_val,
                'range_diff': np.sum(upper_b - lower_b),
                'range_per_product_diff': (upper_b - lower_b).tolist(),
                'range_per_product': bounds_b,
                'prices': b_p.tolist()
            }

        # --- ペナルティ法 (penalty) ---
        for name, pen in penalties.items():
            t0_p = time.perf_counter()
            pen_val, bounds_pen = estimate_bounds_penalty_nelder_all(
                range_bounds,
                t_coefs, t_ints,
                h_coefs, h_ints,
                M, config['K'],
                config['r_min'], config['r_max'],
                pen,
                lam1=1.0, lam2=1.0,
            )
            pen_val_opt, pen_p = optimize_fitted_model(M, bounds_pen, coefs_full, ints_full, init)
            t1_p = time.perf_counter()
            diffs = np.array([u - l for l, u in bounds_pen])
            out[name] = {
                'time': t1_p - t0_p,
                'sales_ratio': pen_val_opt / so_val,
                'true_sales_ratio': np.sum(sales_function(pen_p, alpha, beta)) / so_val,
                'range_diff': diffs.sum(),
                'range_per_product_diff': diffs.tolist(),
                'range_per_product': bounds_pen,
                'prices': pen_p.tolist()
            }
        
        # --- R²スコア計算 （各foldのhatモデルによる予測vs真のY） ---


    lr_full = MultiOutputRegressor(LinearRegression()).fit(X, Y)
    preds_full = lr_full.predict(X)
    r2_list_full = []
    for idx in range(M):
        r2_list_full.append(r2_score(Y[:, idx], preds_full[:, idx]))
        out['r2_list'] = r2_list_full

        
    return (f'M{M}_delta{delta}', out)

# --- メイン ---
def main():
    default_output = 'results_optimized_five_step_penalty_one_thousand_for_paper_with_zero_clip.json'
    args = parse_args(default_output)
    config={'M_list':[5],'delta_list':[0.25,0.5,0.75,1.0],'N': 1000,'K':5,'B':100,'r_mean':0.8,'r_std':0.1,'r_min':0.5,'r_max':1.1}
    if args.n is not None:
        config['N'] = args.n
    quantiles=[1.00,0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60]
    boot_k={100:5.0,99:2.576,95:1.96,90:1.645,85:1.440,80:1.282,75:1.150,70:1.036,65:0.935,60:0.842}
    penalties={'ebpa0.5':0.05,'ebpa1':0.10,'ebpa1.5':0.15,'ebpa2':0.20,'ebpa2.5':0.25,'ebpa3':0.30,'ebpa3.5':0.35,'ebpa4':0.40,'ebpa4.5':0.45,'ebpa5':0.50,'ebpa5.5':0.55,'ebpa6':0.60}
    fixed_widths = parse_fixed_widths(args.fixed_widths)
    run_existing = not (args.only_fixed or args.only_elastic)
    run_fixed = (not args.only_elastic) or args.only_fixed
    run_elastic = (not args.only_fixed) or args.only_elastic
    methods = build_method_names(quantiles, boot_k, penalties, fixed_widths, run_existing, run_fixed, run_elastic)
    only_additional_mode = args.only_fixed or args.only_elastic
    output_path = resolve_output_path(default_output, only_additional_mode, args.output)
    inferred_paper_path = infer_paper_output_path(config)
    merge_into_path = args.merge_into
    if only_additional_mode and args.output is None:
        output_path = inferred_paper_path
    if merge_into_path is None and only_additional_mode and os.path.exists(inferred_paper_path):
        merge_into_path = inferred_paper_path
    # 初期化
    results = {}
    param_combos = list(itertools.product(config["M_list"], config["delta_list"]))
    for M, delta in param_combos:
        key = f"M{M}_delta{delta}"
        results[key] = {}
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
    # 並列実行
    combos=[(i,M,d) for M,d in itertools.product(config['M_list'],config['delta_list']) for i in range(100)]
    total=len(combos)
    with tqdm_joblib(tqdm(desc="Total Experiments",total=total)):
        outs=Parallel(n_jobs=-1)(
            delayed(run_one)(
                i,
                M,
                d,
                config,
                quantiles,
                boot_k,
                penalties,
                fixed_widths,
                args.fixed_center,
                args.only_fixed,
                args.only_elastic,
            )
            for i,M,d in combos
        )
    # 集計
    for key,out in outs:
        for m,metrics in out.items():
            if m=='r2_list':
                results[key]['r2_list'].extend(metrics)
            else:
                for sub,val in metrics.items():
                    results[key][m][sub].append(val)
    # 保存
    if merge_into_path:
        with open(merge_into_path, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
        merge_method_names = methods
        if only_additional_mode:
            merge_method_names = build_additional_method_names(fixed_widths, run_fixed, run_elastic)
        results_to_save = merge_results(existing_results, results, merge_method_names)
        output_path = args.output or merge_into_path
    else:
        results_to_save = results

    with open(output_path,'w', encoding='utf-8') as f:
        json.dump(results_to_save,f,indent=2)

if __name__=='__main__':
    start_time = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start_time
    print(f"Total elapsed time: {elapsed:.2f} seconds")
    print("All experiments completed.")
