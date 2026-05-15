import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import recall_score
import lightgbm as lgb
import joblib
import json
import os
from modeling_f2 import train_improved_autoencoder

def re_train_models():
    print("🚀 [재학습] 데이터 분포 정렬 및 모델 재학습을 시작합니다.")
    
    # 1. 데이터 및 Gap 정보 로드
    train_path = 'data/train_v6_80.csv'
    gap_path = 'validation/results/gap_sensors.json'
    
    if not os.path.exists(train_path) or not os.path.exists(gap_path):
        print("❌ 필요한 파일(train_augmented.csv 또는 gap_sensors.json)이 없습니다.")
        return

    df = pd.read_csv(train_path)
    df.columns = df.columns.str.strip()  # [추가] 컬럼명 공백 제거
    
    with open(gap_path, 'r') as f:
        gap_info = json.load(f)
    # [추가] Gap 정보의 키(센서명) 공백 제거
    gap_info = {k.strip(): v for k, v in gap_info.items()}
    
    print(f"📋 학습 데이터 로드 완료 ({len(df)}건). Gap 센서 {len(gap_info)}개 발견.")

    # 2. [핵심] 데이터 분포 정렬 (Distribution Realignment)
    print(f"📊 {len(gap_info)}개 센서에 대해 데이터 정렬을 시작합니다...")
    for i, (sensor, stats) in enumerate(gap_info.items()):
        if sensor in df.columns:
            real_mean = stats['Real_Mean']
            real_std = stats['Real_Std']
            synth_mean = stats['Synth_Mean']
            synth_std = stats['Synth_Std']
            
            idx = df['Is_Synthetic'] == 1
            # 보정 공식 적용
            df.loc[idx, sensor] = (df.loc[idx, sensor] - synth_mean) * (real_std / (synth_std + 1e-9)) + real_mean
        
        if (i+1) % 20 == 0:
            print(f"   - {i+1}개 센서 정렬 완료...")
    
    print("✅ 데이터 분포 정렬 완료.")

    # 3. 전처리 (Scaling) & 시계열 특성(Rolling) 추출
    print("🧹 데이터 전처리 및 시계열 특징 추출 중...")
    exclude = ['Time_Step', 'Time', 'Step Number', 'Run_Name', 'run_id', 'Fault_Name', 'Is_Synthetic', 'Synthesis_Method', 'Data_Type', 'TIME', 'Time.1', 'TIME.1']
    base_features = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]
    
    window_size = 10
    print(f"🔄 Rolling Window (size={window_size}) 기반 특징 추출...")
    # Run_Name 단위로 독립적인 시계열 롤링 연산 (데이터 섞임 방지)
    rolling_mean = df.groupby('Run_Name')[base_features].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True).sort_index()
    rolling_std = df.groupby('Run_Name')[base_features].rolling(window=window_size, min_periods=1).std().fillna(0).reset_index(level=0, drop=True).sort_index()
    
    # Lag 특성 추출
    lag_1 = df.groupby('Run_Name')[base_features].shift(1)
    lag_1 = lag_1.fillna(df[base_features])
    
    lag_2 = df.groupby('Run_Name')[base_features].shift(2)
    lag_2 = lag_2.fillna(lag_1)
    
    X = np.hstack([df[base_features].values, rolling_mean.values, rolling_std.values, lag_1.values, lag_2.values])
    y_labels = df['Fault_Name'].values
    features = base_features # Save base feature names to reconstruct during inference
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)
    
    # 4. 오토인코더(AE) 학습 (정상 데이터만 사용)
    X_normal = X_scaled[df['Fault_Name'] == 'Normal']
    
    print(f"🏋️  오토인코더(Improved) 학습 시작 (데이터 수: {len(X_normal)})...")
    model_ae, new_threshold = train_improved_autoencoder(
        X_normal_scaled=X_normal,
        input_dim=X_scaled.shape[1],
        epochs=100,
        batch_size=256,
        lr=1e-3,
        dropout=0.1,
        percentile=99.0
    )
    suspect_threshold = new_threshold * 0.8  # 하위 호환성을 위해 유지


    # 6. LightGBM 학습
    print("🏋️  LightGBM 학습 시작 (Sklearn API)...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        objective='multiclass',
        random_state=42,
        is_unbalance=True,  # [추가] 클래스 불균형 해소
        verbose=-1
    )
    lgb_model.fit(X_scaled, y_encoded)

    # 7. 모델 및 상태 저장
    os.makedirs('models_final', exist_ok=True)
    # 모델 저장 시 suspect_threshold 추가
    torch.save({
        'model_state_dict': model_ae.state_dict(),
        'threshold': float(new_threshold),
        'suspect_threshold': float(suspect_threshold),
        'features': features
    }, 'models_final/autoencoder.pth')
    
    joblib.dump(scaler, 'models_final/scaler.joblib')
    joblib.dump(lgb_model, 'models_final/lightgbm_model.joblib')
    joblib.dump(le, 'models_final/label_encoder.joblib')
    
    print("💾 모든 모델 및 전처리기가 models_final/ 폴더에 저장되었습니다.")

if __name__ == "__main__":
    re_train_models()
