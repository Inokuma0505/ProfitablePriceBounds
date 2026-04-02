import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray
from src.price_optimization.price_optimization import predict_optimize


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