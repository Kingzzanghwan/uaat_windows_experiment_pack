\# UAAT Policy Card



\- Base score: features.csv의 score 열을 사용한다.

\- Uncertainty: features.csv의 uncertainty 열을 사용한다. E1은 entropy/TTA instability, E2/E3는 해당 feature extraction 단계에서 저장한 uncertainty를 사용한다.

\- Context: ctx\_\* 열을 사용한다. 밝기, 대비, 블러, 채도 등 입력 상태를 나타내는 feature다.

\- Policy output: sample-specific threshold tau(x).

\- Decision rule: score > tau(x)이면 자동 판단, 아니면 defer.

\- Baselines:

&#x20; - fixed: 전체 샘플에 동일한 global threshold를 사용하는 방식.

&#x20; - uncertainty\_grid: uncertainty에 따라 threshold를 grid로 조정하는 방식.

&#x20; - uaat\_monotone: score, uncertainty, context를 이용하고, uncertainty가 커질 때 threshold가 낮아지지 않도록 제한하는 방식.

\- Risk: C\_wrong \* wrong\_auto\_rate + C\_defer \* defer\_rate.

\- Leakage rule: tau 학습과 coverage 보정은 train/calibration 데이터에서만 수행하고, test 데이터는 최종 risk 측정에만 사용한다.



