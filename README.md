# KOSPI-VaR-Backtesting-Project
KOSPI-VaR-Backtesting-Project by JAEMIN KO

# KOSPI VaR Backtesting

KOSPI 지수를 대상으로 세 가지 VaR 모형을 구현하고, 통계적 검정을 통해
모형의 예측력을 검증하는 프로젝트.

## 진행 상황
- [o]  — 1.데이터 수집 및 수익률 분포 분석
- [ ]  — 2.Historical / Parametric / Monte Carlo VaR 구현
- [ ]  — 3.Kupiec POF, Christoffersen, 바젤 트래픽라이트 검정
- [ ]  — 4.위기구간 분석 및 Expected Shortfall

## 1 결과

분석 대상: KOSPI 종합지수 (^KS11), 2015-01 ~ 2024-12, 2,453 관측치

| 항목 | 값 |
|---|---|
| 연율화 변동성 | 16.80% |
| 왜도 | -0.44 |
| 초과첨도 | 8.42 |
| Jarque-Bera 통계량 | 7,295.75 (p < 0.001) |

정규성 가설은 1% 유의수준에서 기각되었다. 초과첨도 8.42는 정규분포
대비 꼬리가 현저히 두꺼움을 의미하며, 관측 기간 중 최대 하락일인
2024-08-05(-9.18%)은 정규분포 가정 하에서 8.68σ에 해당한다.
이는 정규분포를 가정한 Parametric VaR이 극단적 손실을 과소평가할
가능성을 시사한다.

![Return Analysis](figures/1_returns_analysis.png)

## 데이터

KRX 정보데이터시스템의 회원제 전환으로 pykrx 사용이 제한되어
Yahoo Finance API로 전환. 수정종가 기준 일별 데이터.

## 실행

```
pip install -r requirements.txt
python 01_data_exploration.py
```

