import numpy as np

from numpy.typing import NDArray

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


def create_data(M, N, r_mean, r_std, delta=0.1):

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