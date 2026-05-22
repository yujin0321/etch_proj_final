import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import json

# 앞서 작성한 추론 엔진 모듈 임포트 (파일명에 맞게 수정 필요 시 수정)
from modeling_3 import InferenceEngine

print("🚀 모델 평가를 위한 시뮬레이션을 시작합니다...")

# 1. 테스트 데이터 로드
df_test = pd.read_csv('data/test_v6_20.csv')
df_test.columns = df_test.columns.str.strip()

# 2. 합성 데이터(Synthetic Data) 스케일 캘리브레이션 (경로 버그 수정 완료)
json_path = 'validation/results/gap_sensors.json'
if os.path.exists(json_path):
    try:
        with open(json_path, 'r') as f:
            gap_info = json.load(f)
        gap_info = {k.strip(): v for k, v in gap_info.items()}
        for sensor, stats in gap_info.items():
            if sensor in df_test.columns:
                idx = df_test['Is_Synthetic'] == 1
                df_test.loc[idx, sensor] = (df_test.loc[idx, sensor] - stats['Synth_Mean']) * (stats['Real_Std'] / (stats['Synth_Std'] + 1e-9)) + stats['Real_Mean']
    except Exception as e:
        print(f"⚠️ JSON 캘리브레이션 중 오류 발생 (무시하고 진행): {e}")

# 3. 추론 엔진 초기화
engine = InferenceEngine(window_size=10, model_dir='models_final', use_improved_ae=True)
features = engine.features

# 4. 결과 저장을 위한 리스트 초기화
y_true = []
ae_preds = []
ae_mse_scores = [] 
lgbm_preds = []
final_preds = []

current_run = None
known_faults = set(engine.le.classes_)

# 5. 스트리밍 시뮬레이션 진행
for idx, row in df_test.iterrows():
    # Run 단위로 시계열 버퍼 초기화 (과거 런의 데이터가 현재 런에 영향 주지 않도록)
    if 'Run_Name' in df_test.columns:
        if current_run != row['Run_Name']:
            engine.reset()
            current_run = row['Run_Name']
            
    metrics_dict = {feat: row[feat] for feat in features}
    res = engine.predict(metrics_dict)
    
    # 정답 라벨 처리 (학습되지 않은 결함은 UNKNOWN FAULT로 매핑)
    true_fault = row['Fault_Name'] if row['Fault_Name'] in known_faults else 'UNKNOWN FAULT'
    y_true.append(true_fault)
    
    # AE 예측 결과 및 MSE 기록
    ae_pred = 1 if res['ae_anomaly'] else 0
    ae_preds.append(ae_pred)
    ae_mse_scores.append(res['mse']) 
    
    # LightGBM 단독 예측 및 최종 융합 예측 기록
    lgbm_preds.append(res['predicted_label'])
    final_preds.append(res['status'])

# 이상 탐지(이진 분류)용 정답 라벨 (Normal=0, 나머지=1)
binary_true = [0 if y == 'Normal' else 1 for y in y_true]

# ==============================================================================
# 📊 [평가 결과 출력]
# ==============================================================================

print("\n" + "="*60)
print("📊 1. [Autoencoder 성능 평가: 정상 vs 비정상 (이진 분류)]")
print("="*60)
acc_ae = accuracy_score(binary_true, ae_preds)
rec_ae = recall_score(binary_true, ae_preds)
f1_ae = f1_score(binary_true, ae_preds)

try:
    auc_ae = roc_auc_score(binary_true, ae_mse_scores)
except ValueError:
    auc_ae = 0.0 # 테스트셋에 정상 혹은 비정상 한 종류만 있을 경우 에러 방지

print(f"✔️ AE 정확도(Accuracy)  : {acc_ae*100:.2f}%")
print(f"✔️ AE 재현율(Recall)    : {rec_ae*100:.2f}%")
print(f"✔️ AE F1-스코어         : {f1_ae:.4f}")
print(f"✔️ AE ROC-AUC Score   : {auc_ae:.4f} (임계치와 무관한 본연의 탐지력)")


print("\n" + "="*60)
print("📊 2. [LightGBM 성능 평가: 다중 분류 (AE 판단 제외)]")
print("="*60)
acc_lgbm = accuracy_score(y_true, lgbm_preds)
rec_lgbm = recall_score(y_true, lgbm_preds, average='macro', zero_division=0)
macro_f1_lgbm = f1_score(y_true, lgbm_preds, average='macro', zero_division=0)

print(f"✔️ LightGBM 정확도(Accuracy)  : {acc_lgbm*100:.2f}%")
print(f"✔️ LightGBM 재현율(Macro Rec) : {rec_lgbm*100:.2f}%")
print(f"✔️ LightGBM F1-스코어(Macro)  : {macro_f1_lgbm:.4f}")
print("\n[LightGBM 상세 리포트 요약]")
print(classification_report(y_true, lgbm_preds, zero_division=0))


print("\n" + "="*60)
print("📊 3. [최종(Final) 성능 평가: AE + LightGBM 융합 로직 적용]")
print("="*60)
acc_final = accuracy_score(y_true, final_preds)
rec_final = recall_score(y_true, final_preds, average='macro', zero_division=0)
macro_f1_final = f1_score(y_true, final_preds, average='macro', zero_division=0)

print(f"✔️ 최종 융합 정확도(Accuracy)  : {acc_final*100:.2f}%")
print(f"✔️ 최종 융합 재현율(Macro Rec) : {rec_final*100:.2f}%")
print(f"✔️ 최종 융합 F1-스코어(Macro)  : {macro_f1_final:.4f}")
print("\n[최종 상세 리포트 요약]")
print(classification_report(y_true, final_preds, zero_division=0))