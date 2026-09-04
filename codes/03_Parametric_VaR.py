"""Parametric 방식의 VaR 산출: 정규분포 및 Student-t 분포 가정.

저장소 루트에서 실행한다:
    python codes/03_Parametric_VaR.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


DATA_PATH = "data/kospi_returns.csv"
HISTORICAL_PATH = "data/var_historical.csv"
FIGURE_DIR = "figures"
DATA_DIR = "data"
FIGURE_NAME = "03_parametric_var_image.png"
OUTPUT_NAME = "var_parametric.csv"

WINDOW = 250
CONFIDENCE_LEVEL = 0.99

UPDATE_FREQ = 20   # 자유도 갱신 주기(거래일)
NU_MIN = 4.0       # 첨도 발산 및 분위수 비단조 구간 차단
NU_MAX = 30.0      # 이 이상은 정규분포와 실질적 차이 없음


def load_returns(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df["log_return"]


def parametric_var_normal(returns, window=WINDOW, confidence=CONFIDENCE_LEVEL):
    """정규분포 가정 하의 VaR: -(μ + z·σ), z = -2.326 at 99%.

    shift(1)은 look-ahead bias를 차단한다. t시점 VaR은 t-1까지의
    정보만으로 추정되어야 한다.
    """
    z = stats.norm.ppf(1 - confidence)
    mu = returns.rolling(window).mean().shift(1)
    sigma = returns.rolling(window).std().shift(1)
    return -(mu + z * sigma)


def estimate_degrees_of_freedom(returns, window=WINDOW,
                                update_freq=UPDATE_FREQ,
                                nu_min=NU_MIN, nu_max=NU_MAX):
    """확장 윈도우 최대우도추정으로 t분포 자유도를 추정한다.

    롤링 250일 추정은 표본이 정규분포에 가까운 안정 구간에서 ν가 발산하여
    (수만 단위) 사실상 정규분포로 퇴화한다. 확장 윈도우는 과거 관측치를
    누적하므로 극단값 정보가 유지되어 추정이 안정적이다.

    ν는 꼬리 두께를 결정하는 구조적 모수로 변동성보다 느리게 변하므로
    update_freq 간격으로만 갱신한다.

    [nu_min, nu_max] 클리핑의 근거:
      ν < 4에서는 첨도가 발산하며, 표준화 분위수가 ν에 대해 비단조가 되어
      ν가 작을수록 VaR이 오히려 작아지는 역전이 발생한다.
      ν > 30에서는 정규분포와 실질적 차이가 없다.
    """
    values = returns.to_numpy()
    nu = pd.Series(np.nan, index=returns.index)

    current = np.nan
    for i in range(window, len(values)):
        if (i - window) % update_freq == 0:
            sample = values[:i]
            sample = sample[~np.isnan(sample)]
            current = stats.t.fit(sample)[0]
        nu.iloc[i] = current

    return nu.clip(nu_min, nu_max)


def parametric_var_t(returns, window=WINDOW, confidence=CONFIDENCE_LEVEL):
    """Student-t 분포 가정 하의 VaR.

    자유도 ν인 t분포의 분산은 ν/(ν-2)이므로, σ를 척도로 사용하기 전에
    분산이 1이 되도록 분위수를 표준화한다. 이 보정을 생략하면 변동성이
    이중으로 반영되어 VaR이 과대 산출된다.
    """
    mu = returns.rolling(window).mean().shift(1)
    sigma = returns.rolling(window).std().shift(1)
    nu = estimate_degrees_of_freedom(returns).shift(1)

    t_quantile = stats.t.ppf(1 - confidence, nu)
    t_standardized = t_quantile / np.sqrt(nu / (nu - 2))

    return -(mu + t_standardized * sigma), nu


def flag_violations(returns, var):
    valid = var.notna()
    hit = pd.Series(np.nan, index=returns.index)
    hit[valid] = (returns[valid] < -var[valid]).astype(int)
    return hit


def summarize(name, var, hit):
    n = int(hit.notna().sum())
    x = int(hit.sum())

    print(f"{name}")
    print(f"  Testable days   : {n:,}")
    print(f"  Violations      : {x}")
    print(f"  Expected        : {n * (1 - CONFIDENCE_LEVEL):.1f}")
    print(f"  Violation rate  : {x / n:.4f}")
    print(f"  Mean VaR        : {var.mean():.4f}")
    print(f"  Min VaR         : {var.min():.4f} ({var.idxmin().date()})")
    print(f"  Max VaR         : {var.max():.4f} ({var.idxmax().date()})")

    dates = hit[hit == 1].index
    if len(dates) > 0:
        by_year = pd.Series(1, index=dates).groupby(dates.year).sum()
        print("  Violations by year")
        for year, count in by_year.items():
            print(f"    {year}: {count:2d}")
    print()


def plot_comparison(returns, var_dict, path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    colors = {"Historical": "darkorange", "Normal": "crimson", "Student-t": "seagreen"}

    axes[0].plot(returns.index, returns, linewidth=0.4,
                 color="lightsteelblue", label="Actual return")
    for name, var in var_dict.items():
        axes[0].plot(var.index, -var, linewidth=1.0,
                     color=colors.get(name), label=f"{name} VaR")
    axes[0].set_title(f"VaR Comparison ({CONFIDENCE_LEVEL:.0%})")
    axes[0].legend(loc="lower left", fontsize=9)
    axes[0].grid(alpha=0.3)

    for name, var in var_dict.items():
        axes[1].plot(var.index, var, linewidth=1.0,
                     color=colors.get(name), label=name)
    axes[1].set_title("VaR Level Over Time")
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()


def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    returns = load_returns(DATA_PATH)

    var_normal = parametric_var_normal(returns)
    hit_normal = flag_violations(returns, var_normal)

    var_t, nu = parametric_var_t(returns)
    hit_t = flag_violations(returns, var_t)

    summarize("Parametric VaR - Normal", var_normal, hit_normal)
    summarize("Parametric VaR - Student-t", var_t, hit_t)

    print("Estimated degrees of freedom")
    print(f"  Mean : {nu.mean():.2f}")
    print(f"  Range: {nu.min():.2f} ~ {nu.max():.2f}")
    print()

    var_dict = {"Normal": var_normal, "Student-t": var_t}
    if os.path.exists(HISTORICAL_PATH):
        historical = pd.read_csv(HISTORICAL_PATH, index_col=0, parse_dates=True)
        var_dict = {"Historical": historical["var_hs"], **var_dict}

    plot_comparison(returns, var_dict, os.path.join(FIGURE_DIR, FIGURE_NAME))

    result = pd.DataFrame({
        "log_return": returns,
        "var_normal": var_normal,
        "hit_normal": hit_normal,
        "var_t": var_t,
        "hit_t": hit_t,
        "nu": nu,
    })
    result.to_csv(os.path.join(DATA_DIR, OUTPUT_NAME), encoding="utf-8-sig")


if __name__ == "__main__":
    main()
