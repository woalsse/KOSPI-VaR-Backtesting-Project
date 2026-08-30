"""Historical Simulation 방식의 VaR 산출 및 위반 판정.

저장소 루트에서 실행한다:
    python codes/02_historical_var.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "data/kospi_returns.csv"
FIGURE_DIR = "figures"
DATA_DIR = "data"
FIGURE_NAME = "02_historical_var_image.png"
OUTPUT_NAME = "var_historical.csv"

WINDOW = 250
CONFIDENCE_LEVEL = 0.99


def load_returns(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df["log_return"]


def historical_var(returns, window=WINDOW, confidence=CONFIDENCE_LEVEL):
    """과거 window일 수익률의 경험적 분위수를 VaR로 사용한다.

    분포를 모수적으로 가정하지 않아 두꺼운 꼬리가 그대로 반영되나,
    표본에 없던 규모의 손실은 반영하지 못한다.

    shift(1)은 look-ahead bias를 차단한다. t시점 VaR은 t-1까지의
    정보만으로 추정되어야 한다.
    """
    alpha = 1 - confidence
    var = returns.rolling(window).quantile(alpha).shift(1)
    return -var


def flag_violations(returns, var):
    """실제 손실이 VaR을 초과한 날을 1로 표시한 hit sequence를 반환한다."""
    valid = var.notna()
    hit = pd.Series(np.nan, index=returns.index)
    hit[valid] = (returns[valid] < -var[valid]).astype(int)
    return hit


def summarize(var, hit):
    n = int(hit.notna().sum())
    n_violations = int(hit.sum())

    print("Historical Simulation VaR")
    print(f"  Window          : {WINDOW} days")
    print(f"  Confidence      : {CONFIDENCE_LEVEL:.0%}")
    print(f"  Testable days   : {n:,}")
    print()
    print(f"  Violations      : {n_violations}")
    print(f"  Expected        : {n * (1 - CONFIDENCE_LEVEL):.1f}")
    print(f"  Violation rate  : {n_violations / n:.4f}")
    print(f"  Expected rate   : {1 - CONFIDENCE_LEVEL:.4f}")
    print()
    print(f"  Mean VaR        : {var.mean():.4f}")
    print(f"  Min VaR         : {var.min():.4f} ({var.idxmin().date()})")
    print(f"  Max VaR         : {var.max():.4f} ({var.idxmax().date()})")
    print()

    dates = hit[hit == 1].index
    if len(dates) > 0:
        by_year = pd.Series(1, index=dates).groupby(dates.year).sum()
        print("  Violations by year")
        for year, count in by_year.items():
            print(f"    {year}: {count:2d}")


def plot_var(returns, var, hit, path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(returns.index, returns, linewidth=0.5,
                 color="steelblue", label="Actual return")
    axes[0].plot(var.index, -var, linewidth=1.0,
                 color="darkorange", label=f"HS VaR ({CONFIDENCE_LEVEL:.0%})")

    violations = hit[hit == 1].index
    axes[0].scatter(violations, returns.loc[violations],
                    color="red", s=12, zorder=5, label="Violation")

    axes[0].set_title("Historical Simulation VaR vs Actual Returns")
    axes[0].legend(loc="lower left", fontsize=9)
    axes[0].grid(alpha=0.3)

    # 위기 국면 이후 250일이 경과하면 해당 손실이 추정 윈도우에서
    # 이탈하며 VaR이 계단식으로 하락한다.
    axes[1].plot(var.index, var, linewidth=1.0, color="darkorange")
    axes[1].set_title("VaR Level Over Time")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()


def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    returns = load_returns(DATA_PATH)
    var = historical_var(returns)
    hit = flag_violations(returns, var)

    summarize(var, hit)
    plot_var(returns, var, hit, os.path.join(FIGURE_DIR, FIGURE_NAME))

    result = pd.DataFrame({
        "log_return": returns,
        "var_hs": var,
        "hit_hs": hit,
    })
    result.to_csv(os.path.join(DATA_DIR, OUTPUT_NAME), encoding="utf-8-sig")


if __name__ == "__main__":
    main()
