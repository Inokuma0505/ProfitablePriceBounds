import json
import numpy as np

def verify_quan_hypothesis(path: str, delta: float):
    """
    quan100がquan95より劣るケースを抽出し、
    仮説（探索範囲と予測誤差）を検証する。
    """
    print(f"\n===== 仮説検証開始: {path} (δ={delta}) =====")
    
    # --- データ読み込み ---
    with open(path, 'r') as f:
        results = json.load(f)

    M = 5 # 商品数
    key = f"M{M}_delta{delta}"
    data = results[key]

    # --- 変数初期化 ---
    failure_cases = 0
    position_errors = 0
    total_discrepancy_ratio_100 = 0
    total_discrepancy_ratio_95 = 0

    n_trials = len(data['quan100']['true_sales_ratio'])

    # --- 全100試行をループ ---
    for i in range(n_trials):
        sqi_100 = data['quan100']['true_sales_ratio'][i]
        sqi_95 = data['quan95']['true_sales_ratio'][i]
        
        # 1. 「quan100がquan95より明らかに失敗した」ケースを定義
        if sqi_100 < sqi_95 * 0.98: # 例: 2%以上性能が低い場合
            failure_cases += 1
            
            # --- 仮説1：位置の問題を検証 ---
            p_opt_100 = np.array(data['quan100']['prices'][i])
            bounds_95 = data['quan_debug_info'][i]['quan95']['bounds']
            
            is_outside = False
            for j in range(M):
                lower_95, upper_95 = bounds_95[j]
                if not (lower_95 <= p_opt_100[j] <= upper_95):
                    is_outside = True
                    break # 一つでも範囲外なら確定
            
            if is_outside:
                position_errors += 1

            # --- 仮説2：精度の問題を検証 ---
            debug_info_100 = data['quan_debug_info'][i]['quan100']
            debug_info_95 = data['quan_debug_info'][i]['quan95']
            
            total_discrepancy_ratio_100 += debug_info_100['sales_discrepancy_ratio']
            total_discrepancy_ratio_95 += debug_info_95['sales_discrepancy_ratio']

    # --- 結果のサマリーを表示 ---
    print(f"全{n_trials}試行中、「quan100が失敗した」ケースは {failure_cases} 回ありました。")
    if failure_cases > 0:
        print("\n--- 仮説1：位置の検証 ---")
        print(f"失敗ケース {failure_cases} 回のうち、{position_errors} 回 ({position_errors/failure_cases:.1%}) で、")
        print("quan100の最適価格は quan95 の探索範囲の外にありました。")

        print("\n--- 仮説2：精度の検証 ---")
        avg_disc_100 = total_discrepancy_ratio_100 / failure_cases
        avg_disc_95 = total_discrepancy_ratio_95 / failure_cases
        print("失敗ケースにおいて、モデルは売上を平均でこれだけ過大評価していました：")
        print(f"  - quan100の場合: {avg_disc_100:.2f} 倍 (予測売上 / 真の売上)")
        print(f"  - quan95の場合: {avg_disc_95:.2f} 倍 (予測売上 / 真の売上)")
    
    print("=" * 60)


if __name__ == '__main__':
    # ご自身の環境に合わせてファイルパスを修正してください
    p_three_hundred = "check_r2_300_debug.json"
    p_one_thousand = "check_r2_1000_debug.json"
    
    # N=300, δ=1.0 のケースを検証
    verify_quan_hypothesis(path=p_three_hundred, delta=1.0)
    
    # (比較用) N=1000, δ=1.0 のケースを検証
    verify_quan_hypothesis(path=p_one_thousand, delta=1.0)