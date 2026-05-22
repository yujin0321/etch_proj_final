# modeling_f2_v2.py
# 머신러닝과 딥러닝을 결합한 핵심 추론 엔진 [고도화 버전]
#
# 고도화 내용:
#   1. AE 역할 재정의 — OOD 전용 격리 (방어 로직 B 기준 상향)
#   2. LGBM·AE 가중치 기반 융합 점수 (AE_WEIGHT=0.2)
#   3. 클래스별 동적 임계치 calibration
#   4. 개선된 Autoencoder (BatchNorm + Dropout)
#   5. 적응형 MSE 임계치 (mse_history 기반 동적 갱신)
# ============================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
from sklearn.metrics import f1_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("InferenceEngine_v2")


# ──────────────────────────────────────────────
# 고도화 4: 개선된 Autoencoder
# BatchNorm + Dropout으로 일반화 성능 향상
# 임계치를 99th percentile로 올려 과탐지율 감소
# ──────────────────────────────────────────────
class ImprovedAutoencoder(nn.Module):
    """개선된 오토인코더: BatchNorm + Dropout + 대칭 Bottleneck 구조"""

    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# 기존 모델 파일 로드 호환성을 위해 원본 클래스도 유지
class Autoencoder(nn.Module):
    """원본 오토인코더 (레거시 모델 파일 호환용)"""

    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),       nn.ReLU(),
            nn.Linear(32, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ──────────────────────────────────────────────
# 메인 추론 엔진
# ──────────────────────────────────────────────
class InferenceEngine:
    """
    병렬 방어 로직 + 가중치 융합 추론 엔진 (고도화 버전)

    주요 파라미터
    -------------
    model_dir : 모델 파일 디렉터리
    lgbm_confidence_threshold : LGBM 단독 판별 신뢰도 하한 (기본 0.90)
    window_size : 시계열 버퍼 크기
    ae_weight : 융합 점수 내 AE 발언권 (기본 0.20 — AUC 0.68 수준)
    ae_intervention_multiplier : 방어 로직 B 개입 MSE 배수 (기본 2.0)
    use_improved_ae : ImprovedAutoencoder 구조 사용 여부
    """

    # ── 고도화 1: AE 개입 기준 상향 (기존 1.0× → 2.0×)
    _AE_INTERVENTION_MULTIPLIER_DEFAULT = 2.0
    # ── 고도화 2: AE 가중치 축소 (AUC 0.68 기반 보수적 설정)
    _AE_WEIGHT_DEFAULT   = 0.20
    _LGBM_WEIGHT_DEFAULT = 0.80
    # OOD 극단 판별 배수 (방어 로직 C, 기존 3.0 유지)
    _EXTREME_MULTIPLIER  = 3.0

    def __init__(
        self,
        model_dir: str = 'models_final',
        lgbm_confidence_threshold: float = 0.90,
        window_size: int = 100,
        ae_weight: float = _AE_WEIGHT_DEFAULT,
        ae_intervention_multiplier: float = _AE_INTERVENTION_MULTIPLIER_DEFAULT,
        use_improved_ae: bool = False,
        class_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.model_dir = model_dir
        self.lgbm_confidence_threshold = lgbm_confidence_threshold
        self.window_size = window_size
        self.ae_weight   = ae_weight
        self.lgbm_weight = 1.0 - ae_weight
        self.ae_intervention_multiplier = ae_intervention_multiplier
        self.use_improved_ae = use_improved_ae

        self.mse_history    = deque(maxlen=window_size)
        self.feature_buffer = deque(maxlen=window_size)
        self.important_features: List[str] = []

        # 고도화 3: 클래스별 임계치 (외부 주입 또는 calibrate_class_thresholds로 설정)
        self.class_thresholds: Dict[str, float] = class_thresholds or {}

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_models()

    # ──────────────────────────────────────────
    # 모델 로드
    # ──────────────────────────────────────────
    def _load_models(self) -> None:
        ae_path = os.path.join(self.model_dir, 'autoencoder.pth')
        ae_data = torch.load(ae_path, map_location=self.device, weights_only=False)

        self.all_features = [f.strip() for f in ae_data['features']]
        self.features = (
            self.all_features if not self.important_features
            else [f for f in self.all_features if f in self.important_features]
        )
        self.base_threshold = ae_data['threshold']

        input_dim = len(self.features) * 5  # 현재 + 평균 + 표준편차 + Lag1 + Lag2

        state_dict = ae_data['model_state_dict']
        # 자동 감지: BatchNorm 계열 키가 있으면 Improved 구조 사용
        has_batchnorm = any("running_mean" in k for k in state_dict.keys())
        
        if self.use_improved_ae or has_batchnorm:
            if not has_batchnorm and self.use_improved_ae:
                logger.warning("use_improved_ae=True이나 체크포인트에 BatchNorm이 없습니다. 강제 시도합니다.")
            self.model_ae = ImprovedAutoencoder(input_dim).to(self.device)
        else:
            self.model_ae = Autoencoder(input_dim).to(self.device)

        self.model_ae.load_state_dict(state_dict)
        self.model_ae.eval()

        self.scaler    = joblib.load(os.path.join(self.model_dir, 'scaler.joblib'))
        self.lgb_model = joblib.load(os.path.join(self.model_dir, 'lightgbm_model.joblib'))
        self.le        = joblib.load(os.path.join(self.model_dir, 'label_encoder.joblib'))
        self.expanded_features = self._expanded_feature_names(self.features)

        logger.info(
            f"모델 로드 완료 | 피처: {len(self.features)}개 | "
            f"확장 피처: {len(self.expanded_features)}개 | "
            f"기본 임계치: {self.base_threshold:.4f} | "
            f"AE weight: {self.ae_weight:.2f}"
        )

    @staticmethod
    def _expanded_feature_names(features: List[str]) -> List[str]:
        suffixes = ["current", "rolling_mean", "rolling_std", "lag_1", "lag_2"]
        return [f"{feature}__{suffix}" for suffix in suffixes for feature in features]

    # ──────────────────────────────────────────
    # 공정(Run) 전환 시 버퍼 초기화
    # ──────────────────────────────────────────
    def reset(self) -> None:
        self.mse_history.clear()
        self.feature_buffer.clear()
        logger.info("버퍼 초기화 완료 (Run 전환)")

    # ──────────────────────────────────────────
    # 고도화 3: 클래스별 최적 임계치 탐색
    # 검증셋(X_val 스케일링 전, y_val 정수 인코딩)으로 호출
    # ──────────────────────────────────────────
    def calibrate_class_thresholds(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        threshold_range: Tuple[float, float] = (0.30, 0.95),
        step: float = 0.05,
    ) -> Dict[str, float]:
        """
        클래스별 F1 최대화 임계치를 탐색하여 self.class_thresholds에 저장.

        Parameters
        ----------
        X_val : 스케일링 전 피처 배열 (N, input_dim)
        y_val : 정수 인코딩 레이블 배열 (N,)
        """
        X_scaled = self.scaler.transform(X_val)
        probs = self.lgb_model.predict_proba(X_scaled)

        result: Dict[str, float] = {}
        low, high = threshold_range

        for class_idx, class_name in enumerate(self.le.classes_):
            best_thresh, best_f1 = 0.5, 0.0
            for thresh in np.arange(low, high + step * 0.5, step):
                preds = (probs[:, class_idx] >= thresh).astype(int)
                true  = (y_val == class_idx).astype(int)
                score = f1_score(true, preds, zero_division=0)
                if score > best_f1:
                    best_f1, best_thresh = score, float(thresh)
            result[class_name] = best_thresh

        self.class_thresholds = result
        logger.info(f"클래스별 임계치 보정 완료: {result}")
        return result

    # ──────────────────────────────────────────
    # 고도화 5: MSE 히스토리 기반 적응형 임계치
    # ──────────────────────────────────────────
    def _adaptive_threshold(self, base_threshold: float) -> float:
        """
        정상 구간 MSE 히스토리로 임계치를 동적 보정.
        히스토리가 충분하면 (평균 + 3σ)와 base_threshold의 가중 평균 사용.
        """
        if len(self.mse_history) < 30:
            return base_threshold

        arr = np.array(self.mse_history)
        dynamic = float(np.mean(arr) + 3.0 * np.std(arr))
        # 동적 값이 base보다 너무 낮아지는 건 방지 (최소 base의 70%)
        dynamic = max(dynamic, base_threshold * 0.70)
        # 가중 평균: 초기에는 base 우선, 히스토리 쌓일수록 dynamic 반영
        weight = min(len(self.mse_history) / self.window_size, 1.0)
        return (1 - weight) * base_threshold + weight * dynamic

    # ──────────────────────────────────────────
    # 고도화 2: 가중치 기반 융합 점수
    # ──────────────────────────────────────────
    def _fusion_score(
        self,
        max_prob: float,
        mse: float,
        current_threshold: float,
        is_lgbm_anomaly: bool,
    ) -> float:
        """
        LGBM 확신도 + AE 정규화 MSE를 가중 합산.

        Returns
        -------
        score : 0~1, 높을수록 이상 가능성 높음
        """
        normalized_mse = mse / current_threshold          # 1.0 = 임계치와 동일
        ae_score   = min(normalized_mse / 3.0, 1.0)       # 3배 이상에서 포화
        lgbm_score = max_prob if is_lgbm_anomaly else (1.0 - max_prob)

        return self.lgbm_weight * lgbm_score + self.ae_weight * ae_score

    # ──────────────────────────────────────────
    # 메인 예측
    # ──────────────────────────────────────────
    def predict(
        self,
        metrics_dict: Dict[str, Any],
        override_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        고도화 병렬 방어 로직 예측.

        Returns
        -------
        dict 키:
            status          : 최종 판별 레이블
            mse             : AE 복원 오차
            current_threshold : 사용된 임계치
            adaptive_threshold : 적응형 임계치 값
            confidence      : LGBM 최대 확률
            fusion_score    : 가중치 융합 점수 (0~1)
            is_anomaly      : 이상 여부 bool
            predicted_label : LGBM 원본 예측
            ae_anomaly      : AE 단독 이상 판별 bool
            top_candidates  : 상위 3개 후보 클래스
        """
        if not metrics_dict:
            return {"status": "ERROR", "message": "입력 데이터가 없습니다."}

        metrics_dict = {k.strip(): v for k, v in metrics_dict.items()}

        try:
            # ── 1. 전처리 및 Lag 피처 생성 ──────────────
            row_list = [float(metrics_dict.get(f, 0.0)) for f in self.features]
            self.feature_buffer.append(row_list)

            buffer_arr = np.array(self.feature_buffer)
            row_mean   = np.mean(buffer_arr, axis=0)
            row_std    = (
                np.std(buffer_arr, axis=0, ddof=1)
                if len(self.feature_buffer) > 1
                else np.zeros_like(row_mean)
            )

            if len(self.feature_buffer) >= 3:
                lag_1, lag_2 = buffer_arr[-2], buffer_arr[-3]
            elif len(self.feature_buffer) == 2:
                lag_1 = lag_2 = buffer_arr[-2]
            else:
                lag_1 = lag_2 = np.array(row_list)

            X_array = np.hstack([row_list, row_mean, row_std, lag_1, lag_2]).reshape(1, -1)
            X_scaled = self.scaler.transform(X_array)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            # ── 2. [병렬 A] LightGBM 예측 ───────────────
            probs    = self.lgb_model.predict_proba(X_scaled)[0]
            max_prob = float(np.max(probs))
            pred_idx = int(np.argmax(probs))
            pred_label = self.le.inverse_transform([pred_idx])[0]

            # 고도화 3: 클래스별 임계치 적용
            if self.class_thresholds and pred_label in self.class_thresholds:
                cls_thresh = self.class_thresholds[pred_label]
                if max_prob < cls_thresh and pred_label != 'Normal':
                    # 신뢰도가 해당 클래스 임계치 미달 → 2위 후보 재검토
                    sorted_idx = np.argsort(probs)[::-1]
                    for alt_idx in sorted_idx[1:4]:
                        alt_label = self.le.inverse_transform([alt_idx])[0]
                        alt_thresh = self.class_thresholds.get(alt_label, 0.5)
                        if float(probs[alt_idx]) >= alt_thresh:
                            pred_label = alt_label
                            pred_idx   = alt_idx
                            max_prob   = float(probs[alt_idx])
                            break

            # ── 3. [병렬 B] Autoencoder MSE 계산 ────────
            with torch.no_grad():
                recon = self.model_ae(X_tensor)
                mse   = torch.mean((X_tensor - recon) ** 2).item()

            # ── 4. 임계치 결정 ───────────────────────────
            # 고도화 5: 적응형 임계치
            adaptive_thresh  = self._adaptive_threshold(self.base_threshold)
            current_threshold = override_threshold if override_threshold is not None else adaptive_thresh
            extreme_threshold = current_threshold * self._EXTREME_MULTIPLIER

            # 고도화 1: AE 개입 기준 상향 (base × 2.0)
            ae_intervention_threshold = current_threshold * self.ae_intervention_multiplier

            # ── 5. 1차 판별 ──────────────────────────────
            is_lgbm_anomaly = (pred_label != 'Normal')
            ae_anomaly      = (mse > current_threshold)
            final_status    = pred_label
            is_anomaly      = is_lgbm_anomaly

            # ── 6. 고도화 융합 방어 로직 ─────────────────

            # [방어 로직 A] 과탐지 방어
            # LGBM이 결함이라는데 MSE가 정상 구간의 절반 이하 + 확신도도 낮으면 → 정상 보정
            if is_anomaly and mse < (current_threshold * 0.5) and max_prob < 0.85:
                final_status = 'Normal'
                is_anomaly   = False

            # [방어 로직 B] 미탐지 방어 (고도화 1: 개입 기준 2.0× 상향)
            # LGBM은 정상, AE MSE가 2× 초과 AND LGBM 확신도도 낮을 때만 개입
            elif not is_anomaly and mse > ae_intervention_threshold:
                if max_prob < 0.80:
                    # 두 신호 모두 의심스러울 때만 UNKNOWN FAULT
                    final_status = 'UNKNOWN FAULT'
                    is_anomaly   = True
                # 확신도 높으면 LGBM 신뢰 → Normal 유지
                else:
                    logger.debug(
                        f"방어B 스킵: MSE={mse:.4f} > intervention={ae_intervention_threshold:.4f} "
                        f"이지만 LGBM 확신도 {max_prob:.3f} >= 0.80 → Normal 유지"
                    )

            # [방어 로직 C] 극단 OOD (학습 분포 완전 이탈)
            if mse > extreme_threshold:
                final_status = 'UNKNOWN FAULT'
                is_anomaly   = True

            # ── 7. 고도화 2: 가중치 융합 점수 계산 ───────
            f_score = self._fusion_score(max_prob, mse, current_threshold, is_lgbm_anomaly)

            # 융합 점수가 매우 낮으면 (정상 신호 강함) is_anomaly 재검토
            if is_anomaly and f_score < 0.25 and final_status != 'UNKNOWN FAULT':
                final_status = 'Normal'
                is_anomaly   = False

            # ── 8. 상위 3개 후보 추출 ────────────────────
            filtered_probs = probs.copy()
            normal_idx     = list(self.le.classes_).index('Normal')

            if is_anomaly:
                filtered_probs[normal_idx] = 0.0
                total = np.sum(filtered_probs)
                if total > 0:
                    filtered_probs /= total

            top_indices    = np.argsort(filtered_probs)[::-1][:3]
            top_candidates = [
                {
                    "label":      self.le.inverse_transform([idx])[0],
                    "confidence": float(filtered_probs[idx]),
                }
                for idx in top_indices
            ]

            # ── 9. 정상 샘플 MSE 히스토리 갱신 (Drift 방지) ──
            if not is_anomaly:
                capped_mse = min(mse, current_threshold * 2.0)
                self.mse_history.append(capped_mse)

            return {
                'status':             final_status,
                'mse':                mse,
                'current_threshold':  current_threshold,
                'adaptive_threshold': adaptive_thresh,
                'confidence':         max_prob,
                'fusion_score':       f_score,
                'is_anomaly':         is_anomaly,
                'predicted_label':    pred_label,
                'ae_anomaly':         ae_anomaly,
                'top_candidates':     top_candidates,
                'scaled_features':    X_scaled,
            }

        except Exception as e:
            logger.error(f"예측 중 예외 발생: {e}", exc_info=True)
            return {"status": "ERROR", "message": str(e)}


# ──────────────────────────────────────────────
# 재학습 헬퍼 — ImprovedAutoencoder 학습 루프
# 기존 학습 코드와 독립적으로 사용 가능
# ──────────────────────────────────────────────
def train_improved_autoencoder(
    X_normal_scaled: np.ndarray,
    input_dim: int,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    dropout: float = 0.1,
    device: Optional[torch.device] = None,
    percentile: float = 99.0,
) -> Tuple[ImprovedAutoencoder, float]:
    """
    정상 데이터로 ImprovedAutoencoder를 학습하고 (모델, 임계치)를 반환.

    Parameters
    ----------
    X_normal_scaled : StandardScaler 적용 완료된 정상 데이터 (N, input_dim)
    percentile      : MSE 임계치 분위수 (99th → 과탐지율 감소)

    Returns
    -------
    model     : 학습된 ImprovedAutoencoder
    threshold : percentile 기반 MSE 임계치
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ImprovedAutoencoder(input_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    dataset = torch.FloatTensor(X_normal_scaled).to(device)
    n = len(dataset)

    logger.info(f"ImprovedAutoencoder 학습 시작 | 샘플: {n} | epochs: {epochs}")

    for epoch in range(epochs):
        model.train()
        idx   = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            batch = dataset[idx[i:i + batch_size]]
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs} | loss: {total_loss/n:.6f}")

    # 임계치: 99th percentile (기존 95th → 과탐지율 감소)
    model.eval()
    with torch.no_grad():
        recon  = model(dataset)
        errors = torch.mean((dataset - recon) ** 2, dim=1).cpu().numpy()
    threshold = float(np.percentile(errors, percentile))
    logger.info(f"학습 완료 | {percentile}th percentile 임계치: {threshold:.6f}")

    return model, threshold


# ──────────────────────────────────────────────
# 간단 동작 테스트
# ──────────────────────────────────────────────
if __name__ == "__main__":
    engine = InferenceEngine(
        ae_weight=0.20,
        ae_intervention_multiplier=2.0,
    )
    print("고도화 엔진 초기화 완료.")
    print(f"  AE weight              : {engine.ae_weight}")
    print(f"  LGBM weight            : {engine.lgbm_weight}")
    print(f"  AE 개입 배수           : {engine.ae_intervention_multiplier}×")
    print(f"  극단 OOD 배수          : {engine._EXTREME_MULTIPLIER}×")
    print(f"  클래스별 임계치 보정   : {'활성화' if engine.class_thresholds else '미설정 (calibrate_class_thresholds 호출 필요)'}")