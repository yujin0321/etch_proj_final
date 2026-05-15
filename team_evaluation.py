"""
새 데이터(completed) 기반 모델 재학습 + 성능평가 스크립트
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score
)
import lightgbm as lgb
import joblib
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from inference import Autoencoder

def train_models():
    print("=" * 60)
    print("🚀 [Step 1] 모델 재학습 시작")
    print("=" * 60)

    # 데이터 경로 수정 (현재 폴더에 있는 이름으로 매칭)
    train_path = 'data/train_tstr.csv' 
    df = pd.read_csv(train_path)
    df.columns = df.columns.str.strip()
    
    exclude = ['Time_Step', 'Time', 'Step Number', 'Run_Name', 'run_id',
               'Fault_Name', 'Is_Synthetic', 'Synthesis_Method', 'Data_Type',
               'TIME', 'Time.1', 'TIME.1']
    features = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]

    X = df[features].values
    y_labels = df['Fault_Name'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)

    X_normal = X_scaled[df['Fault_Name'] == 'Normal']
    X_train_tensor = torch.FloatTensor(X_normal)
    input_dim = X_train_tensor.shape[1]

    model_ae = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_ae.parameters(), lr=0.001)

    model_ae.train()
    for epoch in range(30):
        optimizer.zero_grad()
        output = model_ae(X_train_tensor)
        loss = criterion(output, X_train_tensor)
        loss.backward()
        optimizer.step()

    model_ae.eval()
    with torch.no_grad():
        recon = model_ae(X_train_tensor)
        mse = torch.mean((X_train_tensor - recon) ** 2, dim=1).numpy()
        threshold = float(np.percentile(mse, 90))
        suspect_threshold = float(np.percentile(mse, 70))

    lgb_model = lgb.LGBMClassifier(
        n_estimators=100, learning_rate=0.1,
        objective='multiclass', random_state=42,
        is_unbalance=True, verbose=-1
    )
    lgb_model.fit(X_scaled, y_encoded)

    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model_ae.state_dict(),
        'threshold': threshold,
        'suspect_threshold': suspect_threshold,
        'features': features
    }, 'models/autoencoder.pth')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(lgb_model, 'models/lightgbm_model.joblib')
    joblib.dump(le, 'models/label_encoder.joblib')

    stats = {}
    normal_df = df[df['Fault_Name'] == 'Normal']
    for f in features:
        vals = normal_df[f]
        mean_val = float(vals.mean())
        std_val = float(vals.std())
        stats[f] = {
            'mean': mean_val,
            'std': std_val,
            'upper_bound': mean_val + 2 * std_val,
            'lower_bound': mean_val - 2 * std_val
        }
    with open('models/sensor_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    return features, le


def evaluate_models():
    print("=" * 60)
    print("🔍 [Step 2] 성능평가 시작")
    print("=" * 60)

    from inference import InferenceEngine
    engine = InferenceEngine()

    test_path = 'data/Augmented_Sensor_Data_v6_completed.csv'
    df = pd.read_csv(test_path)
    df.columns = df.columns.str.strip()

    y_true_anomaly, y_pred_anomaly = [], []
    y_true_label, y_pred_label = [], []
    mse_scores, threshold_history = [], []

    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        metrics = row.to_dict()
        true_label = metrics['Fault_Name']
        is_true_anomaly = (true_label != 'Normal')

        result = engine.predict(metrics)
        if result.get('status') == 'ERROR':
            continue

        y_true_anomaly.append(int(is_true_anomaly))
        y_pred_anomaly.append(int(result['is_anomaly']))
        y_true_label.append(true_label)
        y_pred_label.append(result['status'])

    labels = sorted(list(set(y_true_label) | set(y_pred_label)))
    acc = accuracy_score(y_true_label, y_pred_label)
    f1_macro = f1_score(y_true_label, y_pred_label, average='macro', zero_division=0)
    
    print(f"\n전체 정확도(Accuracy): {acc:.4f}")
    print(f"Macro F1-Score:      {f1_macro:.4f}")
    print("\n[상세 분류 리포트]")
    report_str = classification_report(y_true_label, y_pred_label, zero_division=0)
    print(report_str)

if __name__ == "__main__":
    train_models()
    evaluate_models()
