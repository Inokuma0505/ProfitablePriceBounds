import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def create_fig_from_json_final(
    path: str,
    M: int,
    d: int,
    fig_title_label: str,
    ebpa_k1_from_right: int,
    ebpa_k2_from_right: int,
    output_dir: str = "../figures",
):
    """
    JSON内の各キー（deltaごと）で別々の図を作成する。

    fixedXX 系も追加で描画する。凡例名は "Fixed width" とし、
    quan / boot / ebpa とは別の色・マーカーを使う。
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)

    color_dict = {
        "quan": "tab:red",
        "boot": "tab:orange",
        "ebpa": "tab:green",
        "fixed": "tab:blue",
    }
    marker_dict = {
        "quan": "o",
        "boot": "s",
        "ebpa": "^",
        "fixed": "D",
    }

    delta_title = 25

    for key, data in results.items():
        method_names = []
        range_means = []
        sales_means = []
        range_sems = []
        sales_sems = []
        groups = []

        for mn, metrics in data.items():
            if mn.startswith("quan"):
                grp = "quan"
            elif mn.startswith("boot"):
                grp = "boot"
            elif mn.startswith("ebpa"):
                grp = "ebpa"
            elif mn.startswith("fixed"):
                grp = "fixed"
            else:
                continue

            method_names.append(mn)
            r = np.array(metrics["range_diff"])
            s = np.array(metrics["true_sales_ratio"])
            n_r, n_s = len(r), len(s)
            range_means.append(r.mean() / d)
            sales_means.append(s.mean())
            range_sems.append(r.std(ddof=1) / np.sqrt(n_r) if n_r > 1 else 0.0)
            sales_sems.append(s.std(ddof=1) / np.sqrt(n_s) if n_s > 1 else 0.0)
            groups.append(grp)

        fig, ax = plt.subplots(figsize=(6, 4.5))

        for grp_prefix, ls in [("quan", "-"), ("boot", "-"), ("fixed", "-")]:
            idxs = [i for i, g in enumerate(groups) if g == grp_prefix]
            if not idxs:
                continue

            xs = [range_means[i] for i in idxs]
            ys = [sales_means[i] for i in idxs]
            sems = [sales_sems[i] for i in idxs]
            sorted_points = sorted(zip(xs, ys, sems))
            if not sorted_points:
                continue

            xs_s, ys_s, sems_s = zip(*sorted_points)
            ax.errorbar(
                xs_s,
                ys_s,
                yerr=sems_s,
                fmt="-",
                marker=marker_dict[grp_prefix],
                ecolor=color_dict[grp_prefix],
                color=color_dict[grp_prefix],
                capsize=3,
                alpha=0.7,
                linestyle=ls,
            )

        idxs_e = [i for i, g in enumerate(groups) if g == "ebpa"]
        if idxs_e:
            xs_e = [range_means[i] for i in idxs_e]
            ys_e = [sales_means[i] for i in idxs_e]
            sems_e = [sales_sems[i] for i in idxs_e]
            pairs_e = sorted(zip(xs_e, ys_e, sems_e))
            if pairs_e:
                xs_e_s, ys_e_s, sems_e_s = zip(*pairs_e)
                n = len(xs_e_s)
                p1 = max(0, n - ebpa_k1_from_right)
                p2 = max(0, n - ebpa_k2_from_right)
                low, high = sorted([p1, p2])
                high = min(high, n - 1)
                if low <= high:
                    seg_idxs = list(range(low, high + 1))
                    seg_x = [xs_e_s[i] for i in seg_idxs]
                    seg_y = [ys_e_s[i] for i in seg_idxs]
                    seg_sems = [sems_e_s[i] for i in seg_idxs]
                    if seg_x:
                        ax.errorbar(
                            seg_x,
                            seg_y,
                            yerr=seg_sems,
                            fmt="-",
                            marker=marker_dict["ebpa"],
                            ecolor=color_dict["ebpa"],
                            color=color_dict["ebpa"],
                            capsize=3,
                            linewidth=1.5,
                            alpha=0.7,
                        )

        ax.set_xlabel(r"Average price range width")
        ax.set_ylabel(r"Relative revenue")
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker=marker_dict["quan"],
                color=color_dict["quan"],
                linestyle="",
                label="Quantile",
            ),
            Line2D(
                [0],
                [0],
                marker=marker_dict["boot"],
                color=color_dict["boot"],
                linestyle="",
                label="Bootstrap",
            ),
            Line2D(
                [0],
                [0],
                marker=marker_dict["ebpa"],
                color=color_dict["ebpa"],
                linestyle="",
                label="Cross-validation",
            ),
            Line2D(
                [0],
                [0],
                marker=marker_dict["fixed"],
                color=color_dict["fixed"],
                linestyle="",
                label="Fixed width",
            ),
        ]

        match = re.search(r"delta([0-9\.]+)", key)
        delta_val = float(match.group(1)) if match else -1.0

        if M == 300 and np.isclose(delta_val, 1.0):
            leg = ax.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize="16")
        elif M == 300 and np.isclose(delta_val, 0.75):
            leg = ax.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize="16")
        else:
            leg = ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize="16")

        frame = leg.get_frame()
        frame.set_linewidth(1.0)
        frame.set_edgecolor("black")
        frame.set_facecolor("white")

        ax.set_xlim(0.0 - 0.05 * 0.6, 0.6 + 0.05 * 0.6)
        ax.set_ylim(0.85 - 0.05 * (1.0 - 0.85), 1.0 + 0.05 * (1.0 - 0.85))
        ax.set_xticks(np.arange(0.0, 0.6 + 1e-8, 0.1))
        ax.set_yticks(np.arange(0.85, 1.0 + 1e-8, 0.03))
        ax.grid(True)

        plt.tight_layout()

        outfile = os.path.join(output_dir, f"{fig_title_label}_{delta_title}.eps")
        fig.savefig(outfile, format="eps", bbox_inches="tight")
        print(f"Saved EPS: {outfile}")

        plt.show()
        plt.close(fig)

        delta_title += 25
