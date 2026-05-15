# 머신러닝과 딥러닝을 결합하여 센서 데이터의 '이상 여부'와 고장 종류를 판별하는 핵심 추론 엔진
# 1단계로는 오토인코더를 통해 이상치를 탐지하고, 2단계로 LightGBM을 통해 세부 분류 수행

# [고도화 반영 사항]
# 1. 시계열(Lag) 특성 반영: t-1, t-2 시점의 데이터를 피처로 추가하여 궤적(Trajectory) 학습 지원
# 2. 불균형 데이터 대응: 본 추론 엔진에 로드되는 LightGBM 모델 학습 시 is_unbalance=True 적용 권장 (주석 참고)
# 3. SHAP 기반 노이즈 제거: 불필요한 하위 변수를 과감히 제외하고 중요 변수만 활용하도록 필터링 로직 추가

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import logging
from typing import Dict, Any, List
from collections import deque 

# 에러 기록 및 시스템 상태 모니터링을 위한 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("InferenceEngine")

class Autoencoder(nn.Module):
    """이상치 탐지를 위한 오토인코더 모델"""
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class InferenceEngine:
    def __init__(self, model_dir='models_v2', lgbm_confidence_threshold=0.90, very_high_threshold=0.95, window_size=100, std_multiplier=2.0):
        self.model_dir = model_dir
        self.lgbm_confidence_threshold = lgbm_confidence_threshold
        self.very_high_threshold = very_high_threshold
        
        # 동적 임계치 알고리즘 (Sliding Window) 세팅
        self.window_size = window_size         
        self.std_multiplier = std_multiplier   
        self.mse_history = deque(maxlen=self.window_size)
        self.feature_buffer = deque(maxlen=self.window_size) 
        
        # [고도화 3] SHAP 분석 결과로 도출된 중요 변수 리스트 (예시: 하위 노이즈 변수 제외)
        # 실제 환경에서는 SHAP 분석 후 도출된 변수명으로 교체해야 합니다.
        self.important_features = [] 
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_models()

    def _load_models(self):
        ae_path = os.path.join(self.model_dir, 'autoencoder.pth')
        ae_data = torch.load(ae_path, map_location=self.device, weights_only=False)
        
        # 모델 학습 시 사용된 전체 피처 로드
        self.all_features = [f.strip() for f in ae_data['features']]
        
        # [고도화 3] 중요 변수 리스트가 지정되지 않았다면 전체 변수 사용, 지정되었다면 필터링
        if not self.important_features:
            self.features = self.all_features
        else:
            self.features = [f for f in self.all_features if f in self.important_features]
            logger.info(f"SHAP 기반 중요 변수 {len(self.features)}개만 추론에 사용합니다.")

        self.base_threshold = ae_data['threshold']
        self.suspect_threshold = ae_data.get('suspect_threshold', self.base_threshold * 0.8)
        
        # [고도화 1] 모델 입력 차원 재계산: 현재(t) + 평균 + 표준편차 + Lag(t-1) + Lag(t-2) = 피처 수 * 5
        input_dim = len(self.features) * 5
        
        self.model_ae = Autoencoder(input_dim).to(self.device)
        self.model_ae.load_state_dict(ae_data['model_state_dict'])
        self.model_ae.eval()
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.joblib'))
        self.lgb_model = joblib.load(os.path.join(self.model_dir, 'lightgbm_model.joblib'))
        self.le = joblib.load(os.path.join(self.model_dir, 'label_encoder.joblib'))

    def reset(self):
        """새로운 공정(Run) 스트림이 시작될 때 버퍼 초기화"""
        self.mse_history.clear()
        self.feature_buffer.clear()

    def predict(self, metrics_dict: Dict[str, Any], override_threshold=None) -> Dict[str, Any]:
        """단일 센서 데이터 이상 탐지 및 분류 (Suspect Zone 포함 5단계 판별 로직)"""
        if not metrics_dict:
            return {"status": "ERROR", "message": "입력 데이터가 없습니다."}
        
        metrics_dict = {k.strip(): v for k, v in metrics_dict.items()}

        try:
            # SHAP 기반으로 선택된 중요 피처만 추출
            row_list = [float(metrics_dict.get(f, 0.0)) for f in self.features]
            self.feature_buffer.append(row_list)
            
            buffer_arr = np.array(self.feature_buffer)
            row_mean = np.mean(buffer_arr, axis=0)
            
            if len(self.feature_buffer) > 1:
                row_std = np.std(buffer_arr, axis=0, ddof=1)
            else:
                row_std = np.zeros_like(row_mean)
                
            # [고도화 1] 시계열 Lag 특성 추출 (t-1, t-2)
            # 버퍼에 데이터가 충분하지 않으면 현재 데이터(row_list)로 패딩
            if len(self.feature_buffer) >= 3:
                lag_1 = buffer_arr[-2]
                lag_2 = buffer_arr[-3]
            elif len(self.feature_buffer) == 2:
                lag_1 = buffer_arr[-2]
                lag_2 = buffer_arr[-2] # 부족한 과거 데이터는 t-1로 대체
            else:
                lag_1 = np.array(row_list)
                lag_2 = np.array(row_list)
                
            # 입력 피처 병합: 현재값 + 평균 + 표준편차 + Lag1 + Lag2
            X_array = np.hstack([row_list, row_mean, row_std, lag_1, lag_2]).reshape(1, -1)
            X_scaled = self.scaler.transform(X_array)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            with torch.no_grad():
                recon = self.model_ae(X_tensor)
                mse = torch.mean((X_tensor - recon)**2).item()
            
            # 1. 임계치 결정
            if override_threshold is not None:
                current_threshold = override_threshold
            else:
                current_threshold = self.base_threshold

            # 2. 분류 모델 예측
            probs = self.lgb_model.predict_proba(X_scaled)[0]
            max_prob = float(np.max(probs))
            pred_idx = int(np.argmax(probs))
            pred_label = self.le.inverse_transform([pred_idx])[0]

            # 3. 판단 로직
            ae_anomaly = mse > current_threshold
            ae_suspect = mse > self.suspect_threshold
            
            if ae_anomaly:
                if max_prob >= 0.5 and pred_label != 'Normal':
                    final_status = pred_label
                    is_anomaly = True
                else:
                    final_status = "UNKNOWN FAULT"
                    is_anomaly = True
                logger.info(f"🚨 Anomaly Detected | MSE: {mse:.4f} | Pred: {pred_label} ({max_prob:.2f}) -> Final: {final_status}")
            elif ae_suspect:
                if pred_label != 'Normal' and max_prob >= 0.6:
                    final_status = pred_label
                    is_anomaly = True
                else:
                    final_status = "Normal"
                    is_anomaly = False
            else:
                if max_prob >= 0.85 and pred_label != 'Normal':
                    final_status = pred_label
                    is_anomaly = True
                else:
                    final_status = "Normal"
                    is_anomaly = False

            # 4. 상위 3개 후보군 추출
            filtered_probs = probs.copy()
            normal_idx = list(self.le.classes_).index('Normal')
            
            if is_anomaly:
                filtered_probs[normal_idx] = 0 
                sum_probs = np.sum(filtered_probs)
                if sum_probs > 0:
                    filtered_probs = filtered_probs / sum_probs
            
            top_indices = np.argsort(filtered_probs)[::-1][:3]
            top_candidates = []
            for idx in top_indices:
                label = self.le.inverse_transform([idx])[0]
                conf = float(filtered_probs[idx])
                top_candidates.append({"label": label, "confidence": conf})

            # 5. Threshold Drift 방지
            if not is_anomaly:
                capped_mse = min(mse, self.base_threshold * 2.0)
                self.mse_history.append(capped_mse)
                
            return {
                'status': final_status,
                'mse': mse,
                'current_threshold': current_threshold,
                'confidence': max_prob,
                'is_anomaly': is_anomaly,
                'predicted_label': pred_label,
                'ae_anomaly': ae_anomaly,
                'top_candidates': top_candidates
            }
            
        except Exception as e:
            logger.error(f"예측 중 예외 발생: {str(e)}")
            return {"status": "ERROR", "message": "내부 서버 오류"}

# =====================================================================
# [고도화 2] 불균형 데이터 대응력 강화 (학습 스크립트 적용 가이드)
# 추론 엔진에 로드되는 lightgbm_model.joblib은 아래와 같이 불균형 데이터
# 가중치가 반영된 상태로 학습되어야 Recall 및 F1-score가 향상됩니다.
# =====================================================================
"""
[학습 시 LightGBM 파라미터 적용 예시]
import lightgbm as lgb
from sklearn.utils.class_weight import compute_class_weight

# 1. 자동 불균형 처리 (가장 간단한 방법)
lgb_model = lgb.LGBMClassifier(
    is_unbalance=True, 
    n_estimators=100,
    random_state=42
)

# 2. Class Weight 직접 부여 (더 정밀한 제어 필요 시)
# classes = np.unique(y_train)
# weights = compute_class_weight('balanced', classes=classes, y=y_train)
# class_weight_dict = dict(zip(classes, weights))
# lgb_model = lgb.LGBMClassifier(
#     class_weight=class_weight_dict,
#     n_estimators=100,
#     random_state=42
# )
"""

if __name__ == "__main__":
    # 고도화된 엔진 테스트
    engine = InferenceEngine(window_size=50) 
    
    logger.info("연속 데이터 스트리밍 테스트 시작 (Lag 변수 포함)...")
    for i in range(10):
        noise = 5.0 if i == 9 else 0.0 
        # 엔진 초기화 시 self.features 수에 맞게 더미 데이터 생성
        sample_data = {f: np.random.rand() + noise for f in engine.features if hasattr(engine, 'features')}
        
        # 모델이 로드되지 않아 features 속성이 없는 테스트 초기화를 방지하기 위한 안전 코드
        if not sample_data:
             sample_data = {f"sensor_{j}": np.random.rand() + noise for j in range(16)}
             
        result = engine.predict(sample_data)
        if result.get("status") != "ERROR":
            print(f"[{i+1}회차] 상태: {result['status']}, MSE: {result['mse']:.4f}, 현재임계치: {result['current_threshold']:.4f}")