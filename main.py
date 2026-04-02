import itertools
import time
import numpy as np
from tqdm import tqdm
import json
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from numpy.typing import NDArray
from scipy.optimize import minimize
from typing import List, Tuple
from sklearn.metrics import r2_score


from src.create_data.create_data import create_data, create_bounds, sales_function
from src.bounds_boot.bounds_boot import bootstrap_bounds
from src.bounds_penalty.bounds_penalty import estimate_bounds_penalty_nelder_all
from src.bounds_quan.bounds_quan import bound_quan
from src.price_optimization.price_optimization import sales_optimize, predict_optimize


def main():
    
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
            alpha, beta, X, Y = create_data(M, config["N"], config["r_mean"], config["r_std"], delta)

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


if __name__ == "__main__":
    main()