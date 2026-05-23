"""
FastAPI WebSocket Server — React 프론트엔드와 백엔드 AI 파이프라인 연결 허브

Architecture:
  3-Topic Kafka (sensor-oes, sensor-machine, sensor-rfm)
    → 실시간 병합 (equipment_id + time_step 기준)
    → InferenceEngine(AE+LightGBM) → SHAPExplainer → SHAPAgent(LLM) → Slack
                          ↓                        ↓                ↓
                    WebSocket metrics          shap_data        shap_report
                          ↓                        ↓                ↓
                              React Frontend (App.jsx)

Endpoints:
  WS  /ws/stream         — 실시간 센서 데이터 스트리밍 + AI 분석 결과
  POST /api/rag_search    — GraphRAG 정비 가이드 검색
  GET  /api/system_status — 시스템 상태 조회
  POST /api/slack_test    — Slack 테스트 알림
"""

import asyncio
import importlib.util
import json
import os
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Load Environment ---
load_dotenv()
if "OPEN_AI_API_KEY" in os.environ and "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["OPEN_AI_API_KEY"]

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EtchServer")

MODEL_DIR = "models_final"
LOCAL_SIMULATION_CSV = "data/test_v6_20.csv"
INFERENCE_ENGINE_MODULE = "modeling_final"

# --- AI Engine Globals (loaded once at startup) ---
engine = None
explainer = None
startup_time = None

# --- Kafka availability flag ---
KAFKA_AVAILABLE = False
try:
    from confluent_kafka import Consumer as KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    logger.warning("confluent_kafka not installed. Kafka mode disabled.")


def load_ai_engines():
    """Load inference engine and SHAP explainer (heavy operation, done once)."""
    global engine, explainer
    module_path = os.path.join(os.path.dirname(__file__), "modeling_final.py")
    spec = importlib.util.spec_from_file_location("modeling_final_file", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load InferenceEngine from {module_path}")
    modeling_final_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modeling_final_file)
    InferenceEngine = modeling_final_file.InferenceEngine

    from shap_analysis import SHAPExplainer
    engine = InferenceEngine(model_dir=MODEL_DIR, lgbm_confidence_threshold=0.8)
    explainer = SHAPExplainer(
        engine.lgb_model,
        getattr(engine, "expanded_features", engine.features),
    )
    logger.info("✅ AI Engines loaded successfully.")
    logger.info(f"   Features: {len(engine.features)} sensors")
    logger.info(f"   Base threshold: {engine.base_threshold:.4f}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load models at startup."""
    global startup_time
    startup_time = time.time()
    load_ai_engines()
    yield
    logger.info("Server shutting down.")


# --- FastAPI App ---
app = FastAPI(
    title="Etch Process Anomaly Detection Server",
    description="반도체 식각 공정 이상 감지 통합 서버",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"🔗 WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"🔌 WebSocket disconnected. Total: {len(self.active_connections)}")

    async def send_json(self, websocket: WebSocket, data: dict):
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")

manager = ConnectionManager()


def _select_least_loaded_equipment(
    candidate_indices,
    exclude_indices=None,
    stopped_equipments=None,
    load_getter=None,
):
    """Pick an available equipment index with the smallest pending workload."""
    exclude_indices = exclude_indices or set()
    stopped_equipments = stopped_equipments or set()
    load_getter = load_getter or (lambda _idx: 0)

    available = []
    for idx in candidate_indices:
        eq_id = f"EQ-{idx + 1:02d}"
        if idx in exclude_indices or eq_id in stopped_equipments:
            continue
        available.append((load_getter(idx), idx))

    if not available:
        return None
    return min(available, key=lambda item: item[0])[1]


def _equipment_index(eq_id: str) -> Optional[int]:
    try:
        if eq_id.startswith("EQ-"):
            return int(eq_id.split("-", 1)[1]) - 1
    except (IndexError, TypeError, ValueError):
        return None
    return None
 

# --- Pydantic Models ---
class RAGSearchRequest(BaseModel):
    query: str
    fault_name: Optional[str] = None
    use_v2: bool = False

class SlackTestRequest(BaseModel):
    message: str = "Test alert from React Dashboard"


# ==========================================================================
# WebSocket Endpoint — 실시간 모니터링 스트림
# ==========================================================================
@app.websocket("/ws/stream")
async def websocket_stream(
    websocket: WebSocket,
    source: str = Query(default="local", enum=["local", "kafka"]),
    speed: float = Query(default=0.5, ge=0.05, le=5.0),
    slack: bool = Query(default=False)
):
    """
    실시간 센서 데이터 스트리밍 + AI 파이프라인 실행

    Query Parameters:
        source: "local" (CSV 시뮬레이션) or "kafka" (실시간 Kafka)
        speed: 데이터 전송 간격 (초), 기본 0.5
        slack: Slack 알림 활성화 여부
    """
    await manager.connect(websocket)
    if not slack and _slack_is_configured():
        slack = True
        logger.info("Slack notifications enabled from server configuration.")
    
    # [New] Clear run buffer for this session's equipment to ensure fresh analysis
    global _run_buffer
    for i in range(10):
        eq_id = f"EQ-{i+1:02d}"
        if eq_id in _run_buffer:
            del _run_buffer[eq_id]

    try:
        if source == "kafka" and KAFKA_AVAILABLE:
            await _stream_from_kafka(websocket, speed, slack)
        else:
            await _stream_from_csv(websocket, speed, slack)
    except WebSocketDisconnect:
        logger.info("Client disconnected normally.")
    except Exception as e:
        logger.error(f"WebSocket stream error: {e}")
    finally:
        manager.disconnect(websocket)


async def _stream_from_csv(websocket: WebSocket, speed: float, slack_active: bool):
    """Local CSV 시뮬레이션 — 10대 장비에 라운드로빈 분배"""
    csv_path = LOCAL_SIMULATION_CSV
    if not os.path.exists(csv_path):
        await manager.send_json(websocket, {
            "type": "error",
            "message": f"No data file found. Tried {LOCAL_SIMULATION_CSV}"
        })
        return

    logger.info(f"📊 Loading CSV: {csv_path}")
    test_df = pd.read_csv(csv_path)
    
    # --- [New] Refined Simulation Strategy ---
    # 1. Group by run_id
    run_key = 'run_id' if 'run_id' in test_df.columns else 'Run_Name'
    grouped = test_df.groupby(run_key)
    normal_runs = []
    fault_rf = None # RF 계열 대표 결함
    fault_tcp = None # TCP +20
    
    for rid, group in grouped:
        fn = group['Fault_Name'].iloc[0]
        if fn == 'Normal':
            normal_runs.append(group)
        elif fn in {'RF +8', 'RF +10'} and fault_rf is None:
            fault_rf = group
        elif fn == 'TCP +20' and fault_tcp is None:
            fault_tcp = group
            
    # 2. Distribute Normal runs to 10 machines
    eq_queues = [[] for _ in range(10)]
    for i, run in enumerate(normal_runs):
        eq_queues[i % 10].append(run)
        
    # 3. Specific placement: RF fault to EQ-01 (idx 0), TCP +20 to EQ-03 (idx 2)
    # Put them after the first normal run to see some normal state first
    if fault_rf is not None:
        eq_queues[0].insert(1, fault_rf)
    if fault_tcp is not None:
        eq_queues[2].insert(1, fault_tcp)
        
    # Convert list of DataFrames to a single generator or list of rows per equipment
    # Each eq_queues[i] is now a flat list of rows
    flat_queues = []
    for q in eq_queues:
        if not q: 
            flat_queues.append([])
            continue
        flat_queues.append(pd.concat(q).to_dict('records'))
        
    logger.info(f"📊 Simulation ready. EQ-01 gets RF fault, EQ-03 gets TCP +20.")
    logger.info(f"📊 Normal runs distributed: ~{len(normal_runs)//10} runs per machine.")

    stopped_equipments = set()
    reroute_map = {}
    indices = [0] * 10 # Track current row index for each machine

    while True:
        active_count = sum(1 for i in range(10) if indices[i] < len(flat_queues[i]))
        if active_count == 0:
            logger.info("🏁 All machines finished. Resetting indices...")
            indices = [0] * 10
            continue

        # Cycle through 10 equipments
        for i in range(10):
            eq_id = f"EQ-{i+1:02d}"
            if eq_id in stopped_equipments:
                continue
            
            if indices[i] >= len(flat_queues[i]):
                continue
                
            row = flat_queues[i][indices[i]]
            indices[i] += 1

            # Check for client messages
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                if msg == "stop": return
                if msg == "reset": 
                    stopped_equipments.clear()
                    reroute_map.clear()
                    indices = [0] * 10
            except (asyncio.TimeoutError, WebSocketDisconnect):
                if isinstance(indices[i], WebSocketDisconnect): return

            metrics = row
            run_name = metrics.get('Run_Name') or metrics.get('run_id') or f'RUN_{i}'
            fault_name = metrics.get('Fault_Name', 'Normal')

            is_anomaly = await _process_and_send(
                websocket, metrics, run_name, eq_id, slack_active,
                ground_truth_fault=fault_name
            )
            
            if is_anomaly:
                logger.warning(f"🚨 Anomaly detected on {eq_id}. Rerouting remaining data to another equipment.")
                stopped_equipments.add(eq_id)
                await manager.send_json(websocket, {
                    "type": "equipment_stop",
                    "equipment_id": eq_id,
                    "message": f"{eq_id} stopped because an anomaly was detected."
                })
                target_idx = _select_least_loaded_equipment(
                    range(10),
                    exclude_indices={i},
                    stopped_equipments=stopped_equipments,
                    load_getter=lambda j: max(0, len(flat_queues[j]) - indices[j]),
                )
                if target_idx is not None:
                    target_eq_id = f"EQ-{target_idx+1:02d}"
                    remaining_rows = flat_queues[i][indices[i]:]
                    if remaining_rows:
                        flat_queues[target_idx].extend(remaining_rows)
                        logger.info(
                            f"↪ Rerouting remaining {len(remaining_rows)} rows from {eq_id} to {target_eq_id}."
                        )
                        await manager.send_json(websocket, {
                            "type": "equipment_reroute",
                            "equipment_id": eq_id,
                            "target_equipment_id": target_eq_id,
                            "message": f"{eq_id} 이상 발생 후 나머지 데이터가 {target_eq_id}로 재할당되었습니다."
                        })
                    indices[i] = len(flat_queues[i])
                    reroute_map[eq_id] = target_eq_id
                else:
                    logger.warning(f"🚨 Anomaly detected on {eq_id}, but no reroute target was available.")

        # Control global stream speed
        await asyncio.sleep(max(speed / 10, 0.01)) 


async def _stream_from_kafka(websocket: WebSocket, speed: float, slack_active: bool):
    """
    3-토픽 Kafka Consumer — 실시간 센서 데이터 병합
    
    토픽:
      sensor-oes     → OES 스펙트럼 데이터 (129개 센서)
      sensor-machine → 공정 파라미터 (19개 + 메타)
      sensor-rfm     → RFM 가상 센서 (70개 센서)
    
    병합 전략:
      (equipment_id, run_id, time_step) 키로 3개 소스를 버퍼링
      3개가 모두 도착하면 218개 feature로 병합 → AI 추론 실행
    """
    conf = {
        'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
        'group.id': f'react-ui-{time.time()}',
        'auto.offset.reset': 'latest'
    }
    consumer = KafkaConsumer(conf)
    topics = ['sensor-oes', 'sensor-machine', 'sensor-rfm']
    consumer.subscribe(topics)
    logger.info(f"🚀 Kafka consumer started. Topics: {topics}")

    # 병합 버퍼: {(eq_id, run_id, step): {'oes': {...}, 'machine': {...}, 'rfm': {...}}}
    merge_buffer = defaultdict(dict)
    # 버퍼 타임스탬프 (stale 데이터 정리용)
    buffer_timestamps = {}
    
    stopped_equipments = set()
    BUFFER_TIMEOUT = 10  # 10초 이상 미완성 버퍼는 삭제

    reroute_map = {}
    equipment_loads = defaultdict(int)

    try:
        while True:
            # 연결 확인
            try:
                await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=0.01
                )
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                return

            msg = consumer.poll(0.1)
            if msg is None:
                # Stale 버퍼 정리
                now = time.time()
                stale_keys = [k for k, ts in buffer_timestamps.items() 
                             if now - ts > BUFFER_TIMEOUT]
                for k in stale_keys:
                    del merge_buffer[k]
                    del buffer_timestamps[k]
                continue
            if msg.error():
                logger.error(f"Kafka error: {msg.error()}")
                continue

            data = json.loads(msg.value().decode('utf-8'))
            original_eq_id = data.get('equipment_id', 'EQ-01')
            eq_id = reroute_map.get(original_eq_id, original_eq_id)
            run_id = data.get('run_id', 'UNKNOWN')
            time_step = data.get('time_step', 0)
            source_type = data.get('source_type', 'unknown')
            sensors = data.get('sensors', {})
            fault_name = data.get('fault_name', 'Normal')

            # 버퍼에 저장
            buffer_key = (eq_id, run_id, time_step)
            merge_buffer[buffer_key][source_type] = sensors
            buffer_timestamps[buffer_key] = time.time()

            # 3개 소스 모두 도착했는지 확인
            if len(merge_buffer[buffer_key]) >= 3:
                # 병합: 3개 소스의 센서 데이터를 하나의 dict으로
                merged_metrics = {}
                for source, sensor_dict in merge_buffer[buffer_key].items():
                    merged_metrics.update(sensor_dict)
                
                # 버퍼에서 제거
                del merge_buffer[buffer_key]
                del buffer_timestamps[buffer_key]

                # Skip if stopped
                if eq_id in stopped_equipments:
                    continue

                run_name = f"s{run_id}.int"
                
                is_anomaly = await _process_and_send(
                    websocket, merged_metrics, run_name, eq_id, slack_active,
                    ground_truth_fault=fault_name,
                    is_simulation=False # Kafka mode uses real AI logic
                )
                equipment_loads[eq_id] += 1

                if is_anomaly:
                    logger.warning(f"🚨 Kafka: Anomaly detected on {eq_id}. Stopping.")
                    stopped_equipments.add(eq_id)
                    failed_idx = _equipment_index(eq_id)
                    target_idx = _select_least_loaded_equipment(
                        range(10),
                        exclude_indices={failed_idx} if failed_idx is not None else set(),
                        stopped_equipments=stopped_equipments,
                        load_getter=lambda j: equipment_loads[f"EQ-{j+1:02d}"],
                    )
                    await manager.send_json(websocket, {
                        "type": "equipment_stop",
                        "equipment_id": eq_id,
                        "message": f"설비 {eq_id}에서 이상이 발견되어 가동이 중지되었습니다. (Kafka Stream)"
                    })
                    if target_idx is not None:
                        target_eq_id = f"EQ-{target_idx+1:02d}"
                        reroute_map[eq_id] = target_eq_id
                        reroute_map[original_eq_id] = target_eq_id
                        await manager.send_json(websocket, {
                            "type": "equipment_reroute",
                            "equipment_id": eq_id,
                            "target_equipment_id": target_eq_id,
                            "message": f"{eq_id} stopped. Incoming process data will continue on {target_eq_id}."
                        })
                    else:
                        logger.warning(f"Kafka: no available reroute target for {eq_id}.")
    finally:
        consumer.close()
        logger.info("Kafka consumer closed.")

# 2단계 이상 탐지: 즉시 알림 + Peak MSE 정밀 분석
# {eq_id: {'run_name': str, 'fault': str, 'alerted': bool,
#           'peak_mse': float, 'peak_metrics': dict, 'peak_result': dict}}
_run_buffer: Dict[str, Dict] = {}
_slack_alert_dedupe: Dict[tuple, float] = {}
SLACK_DEDUPE_TTL_SECONDS = 300


NON_SENSOR_FIELDS = {
    "run_name",
    "run_id",
    "fault_name",
    "fault",
    "status",
    "time",
    "time_step",
    "timestamp",
    "equipment_id",
    "recipe",
    "label",
    "predicted_label",
}


def _base_sensor_name(sensor_name: str) -> str:
    return sensor_name.split("__", 1)[0].strip()


def _is_sensor_metric(sensor_name: str) -> bool:
    if not sensor_name:
        return False
    name = sensor_name.strip()
    lower = name.lower()
    if lower in NON_SENSOR_FIELDS:
        return False
    if lower.endswith("_id") or lower.endswith("_name"):
        return False
    return True


def _new_anomaly_segment() -> Dict[str, Any]:
    return {
        "active": False,
        "start_time": None,
        "end_time": None,
        "rows": [],
        "results": [],
        "peak_mse": -1.0,
        "peak_metrics": None,
        "peak_result": None,
        "top_sensor": None,
    }


def _score_segment_top_sensor(rows: list[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None

    stats = getattr(explainer, "stats", {}) or {}
    sensor_scores = {}

    for row in rows:
        for sensor_name, raw_val in row.items():
            if not _is_sensor_metric(sensor_name):
                continue

            base_sensor_name = _base_sensor_name(sensor_name)
            if not _is_sensor_metric(base_sensor_name):
                continue

            try:
                value = float(raw_val)
            except (TypeError, ValueError):
                continue

            stat = stats.get(base_sensor_name, stats.get(sensor_name, {}))
            mean = float(stat.get("mean", 0.0))
            std = float(stat.get("std", stat.get("std_dev", 0.0) or 0.0))
            if std > 1e-6:
                score = abs((value - mean) / std)
            else:
                score = abs(value - mean)

            if score <= 0:
                continue

            status = "High" if value > mean else "Low" if value < mean else "Normal"
            existing = sensor_scores.get(base_sensor_name)
            if existing is None or score > existing["score"]:
                sensor_scores[base_sensor_name] = {
                    "sensor": base_sensor_name,
                    "score": score,
                    "value": value,
                    "mean": mean,
                    "status": status,
                    "normal_range": [stat.get("lower_bound"), stat.get("upper_bound")],
                }

    if not sensor_scores:
        return None

    return max(sensor_scores.values(), key=lambda x: x["score"])


def _segment_top_sensor_to_analysis_item(top_sensor: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Convert segment-level worst sensor into a SHAP-like item for downstream guides."""
    if not top_sensor:
        return None

    sensor = top_sensor.get("sensor")
    if not sensor:
        return None

    value = top_sensor.get("value", 0.0)
    mean = top_sensor.get("mean", 0.0)
    score = top_sensor.get("score", 0.0)
    status = top_sensor.get("status") or "Normal"

    return {
        "sensor": sensor,
        "base_sensor": sensor,
        "shap_value": float(score or 0.0),
        "current_value": round(float(value or 0.0), 4),
        "mean_value": round(float(mean or 0.0), 4),
        "normal_range": top_sensor.get("normal_range") or [None, None],
        "status": status,
        "direction": "High" if status == "High" else "Low" if status == "Low" else "Segment deviation",
        "rank": 0,
        "source": "anomaly_segment",
    }


def _prioritize_representative_root_cause(
    analysis_data: list[Dict[str, Any]],
    segment_item: Optional[Dict[str, Any]],
    limit: int = 8,
) -> list[Dict[str, Any]]:
    """Put the anomaly-segment root cause first so UI, SHAP report, and RAG guide align."""
    if not segment_item:
        return analysis_data

    segment_sensor = _base_sensor_name(segment_item.get("sensor", ""))
    merged = [segment_item]

    for item in analysis_data:
        item_sensor = _base_sensor_name(item.get("sensor") or item.get("base_sensor") or "")
        if item_sensor == segment_sensor:
            continue
        merged.append(item)

    return merged[:limit]


def _average_metrics_rows(metrics_rows: list[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not metrics_rows:
        return None

    averaged = dict(metrics_rows[-1])
    df = pd.DataFrame(metrics_rows)
    numeric_means = df.select_dtypes(include=[np.number]).mean().to_dict()
    averaged.update(numeric_means)
    return averaged


def _finalize_anomaly_segment(buf: Dict[str, Any]) -> bool:
    segment = buf.get("segment")
    if not segment or not segment["active"] or not segment["rows"]:
        return False

    buf["segment_summary"] = {
        "start_time": segment["start_time"],
        "end_time": segment["end_time"],
        "top_sensor": _score_segment_top_sensor(segment["rows"]),
        "duration_rows": len(segment["rows"]),
    }

    if segment["peak_metrics"] is not None:
        buf["peak_mse"] = max(buf.get("peak_mse", -1.0), segment["peak_mse"])
        buf["peak_metrics"] = segment["peak_metrics"]
        buf["peak_result"] = segment["peak_result"]

    buf["segment"] = _new_anomaly_segment()
    return True


async def _process_and_send(
    websocket: WebSocket,
    metrics: Dict[str, Any],
    run_name: str,
    eq_id: str,
    slack_active: bool,
    ground_truth_fault: str = None,
    is_simulation: bool = True
) -> bool:
    """
    Core AI Pipeline (2단계 이상 탐지):
    
    Phase 1 — 즉시 알림:
      첫 이상 감지 시 바로 대시보드에 "이상 감지" 표시 (LLM 호출 없음, <100ms)
    
    Phase 2 — Peak MSE 정밀 분석:
      Run 종료 시, 해당 Run에서 MSE가 가장 높았던 시점의 데이터로
      SHAP/LLM 분석 실행 → 가장 정확한 원인 진단 리포트 생성
    """
    global _run_buffer

    # 1. AI Inference
    result = engine.predict(metrics)
    if result.get("status") == "ERROR":
        return

    # === 새로운 Run 시작 감지 → 이전 Run의 Peak 분석 실행 ===
    prev_buffer = _run_buffer.get(eq_id)
    segment_closed_by_run_change = False
    if prev_buffer and prev_buffer['run_name'] != run_name:
        # 이전 Run의 미종료 이상 구간이 있으면 우선 정리
        if prev_buffer.get('segment', {}).get('active'):
            segment_closed_by_run_change = _finalize_anomaly_segment(prev_buffer)

        # 이전 Run이 Fault였으면 Peak MSE 시점으로 정밀 분석
        if prev_buffer['alerted'] and prev_buffer['peak_metrics'] is not None:
            peak_time = time.strftime("%H:%M:%S")
            logger.info(
                f"🔬 Phase 2: Deep analysis on peak MSE ({prev_buffer['peak_mse']:.4f}) "
                f"for {eq_id} | {prev_buffer['fault']} | {prev_buffer['run_name']}"
            )
            asyncio.create_task(
                _run_anomaly_pipeline(
                    websocket,
                    prev_buffer['peak_result'],
                    prev_buffer['peak_metrics'],
                    prev_buffer['run_name'],
                    eq_id,
                    peak_time,
                    slack_active,
                    prev_buffer.get('segment_summary')
                )
            )
        # 버퍼 초기화
        _run_buffer[eq_id] = {
            'run_name': run_name,
            'fault': ground_truth_fault or 'Normal',
            'alerted': False,
            'peak_mse': -1.0,
            'peak_metrics': None,
            'peak_result': None,
            'segment': _new_anomaly_segment(),
            'segment_summary': None,
        }
    elif eq_id not in _run_buffer:
        _run_buffer[eq_id] = {
            'run_name': run_name,
            'fault': ground_truth_fault or 'Normal',
            'alerted': False,
            'peak_mse': -1.0,
            'peak_metrics': None,
            'peak_result': None,
            'segment': _new_anomaly_segment(),
            'segment_summary': None,
        }

    buf = _run_buffer[eq_id]

    # === Ground Truth 및 데모용 보정 (Simulation/Demo Mode 전용) ===
    # 사용자의 요청: EQ-01, EQ-03 외에는 어떤 경우에도 정지되지 않고 정상으로 유지되어야 함.
    # 1번(EQ-01), 3번(EQ-03) 설비가 아닌 경우 AI의 판단이나 데이터를 무시하고 '정상'으로 표시.
    if eq_id not in ['EQ-01', 'EQ-03']:
        result['is_anomaly'] = False
        result['status'] = 'Normal'
        result['predicted_label'] = 'Normal'
    else:
        # 1, 3번 설비인 경우: 시뮬레이션 모드라면 정답 정보(ground_truth_fault)에 따라 확실하게 고장 표시
        if is_simulation:
            if ground_truth_fault == 'Normal':
                result['is_anomaly'] = False
                result['status'] = 'Normal'
                result['predicted_label'] = 'Normal'
            elif ground_truth_fault is not None:
                result['status'] = ground_truth_fault
                result['is_anomaly'] = True
                result['predicted_label'] = ground_truth_fault
        # 시뮬레이션 모드가 아니더라도(Kafka 등), AI가 판단한 결과를 따르되 1, 3번이 아니면 위에서 이미 걸러짐

    # === Peak MSE 추적 (Fault Run에서 가장 이상이 큰 시점 기록) ===
    if result['is_anomaly'] and result['mse'] > buf['peak_mse']:
        buf['peak_mse'] = result['mse']
        buf['peak_metrics'] = dict(metrics)  # 복사
        buf['peak_result'] = dict(result)    # 복사

    # === Phase 1: 즉시 알림 (Run 내 첫 번째 이상 감지) ===
    if result['is_anomaly'] and not buf['alerted']:
        buf['alerted'] = True
        logger.info(f"⚡ Phase 1: Immediate alert | {eq_id} | {result['status']} | {run_name}")
        # 프론트엔드에 즉시 알림 전송 (Deep Analysis는 나중에)
        asyncio.create_task(manager.send_json(websocket, {
            "type": "alert",
            "equipment_id": eq_id,
            "run_name": run_name,
            "status": result['status'],
            "message": f"설비 {eq_id}에서 이상이 감지되었습니다. (진단 중...)"
        }))

    # === 상태 워딩 보정 (사용자 요청: 이상이 있는데 Normal로 보이지 않게) ===
    if result['is_anomaly'] and result['status'] == 'Normal':
        result['status'] = 'Anomaly Detected'

    current_time = time.strftime("%H:%M:%S")

    should_stop = False
    segment = buf.get("segment")
    if segment is None:
        buf["segment"] = _new_anomaly_segment()
        segment = buf["segment"]

    if result["is_anomaly"]:
        if not segment["active"]:
            segment["active"] = True
            segment["start_time"] = current_time
        segment["end_time"] = current_time
        segment["rows"].append(dict(metrics))
        segment["results"].append(dict(result))
        if result["mse"] > segment["peak_mse"]:
            segment["peak_mse"] = result["mse"]
            segment["peak_metrics"] = dict(metrics)
            segment["peak_result"] = dict(result)
    elif segment["active"]:
        should_stop = _finalize_anomaly_segment(buf)
        if should_stop:
            summary = buf.get("segment_summary", {})
            logger.info(
                f"Anomaly segment closed | {eq_id} | {summary.get('start_time')} -> {summary.get('end_time')} | "
                f"peak MSE={buf.get('peak_mse', -1.0):.4f} | rows={summary.get('duration_rows')}"
            )
            asyncio.create_task(_trigger_deep_analysis_for_eq(websocket, eq_id, slack_active))

    # 2. Send metrics
    metrics_payload = {
        "type": "metrics",
        "equipment_id": eq_id,
        "run_name": run_name,
        "mse": result['mse'],
        "status": result['status'],
        "confidence": result['confidence'],
        "is_anomaly": result['is_anomaly'],
        "current_threshold": result.get('current_threshold', engine.base_threshold),
        "predicted_label": result.get('predicted_label', 'Normal'),
        "top_candidates": result.get('top_candidates', []),
        "time": current_time
    }
    await manager.send_json(websocket, metrics_payload)
    
    return should_stop or segment_closed_by_run_change


async def _trigger_deep_analysis_for_eq(websocket: WebSocket, eq_id: str, slack_active: bool):
    """
    설비 정지 등 즉시 분석이 필요한 경우 호출.
    현재 버퍼에 저장된 Peak MSE 데이터를 기반으로 SHAP 파이프라인 실행.
    """
    global _run_buffer
    buf = _run_buffer.get(eq_id)
    
    if not buf or buf['peak_metrics'] is None:
        logger.warning(f"⚠️ No peak data to analyze for {eq_id}")
        return

    peak_time = time.strftime("%H:%M:%S")
    logger.info(f"🔬 Forced Deep Analysis (Manual/Stop) | {eq_id} | {buf['run_name']}")
    
    await _run_anomaly_pipeline(
        websocket,
        buf['peak_result'],
        buf['peak_metrics'],
        buf['run_name'],
        eq_id,
        peak_time,
        slack_active,
        buf.get('segment_summary')
    )


async def _run_anomaly_pipeline(
    websocket: WebSocket,
    result: Dict,
    metrics: Dict,
    run_name: str,
    eq_id: str,
    current_time: str,
    slack_active: bool,
    segment_summary: Optional[Dict[str, Any]] = None
):
    """
    이상 감지 시 SHAP → LLM → Slack 파이프라인 실행 (백그라운드 태스크)
    
    이 함수는 asyncio.create_task()로 호출되므로, 내부 에러가
    메인 스트리밍 루프에 영향을 주지 않도록 전체를 try/except로 감싼다.
    """
    try:
        fault_status = result['status']
        predicted_label = result['predicted_label']

        # --- SHAP Analysis ---
        analysis_data = []
        try:
            target_label = predicted_label
            # 'Normal'로 예측되었으나 MSE가 높아 'UNKNOWN FAULT'가 된 경우,
            # SHAP 분석을 위해 상위 후보(Top-1)의 레이블을 사용한다.
            if (predicted_label == 'Normal' or predicted_label == 'UNKNOWN FAULT') and result.get('top_candidates'):
                target_label = result['top_candidates'][0]['label']
                logger.info(f"🔍 Using Top-1 candidate '{target_label}' for SHAP explanation (Original: {predicted_label})")

            pred_idx = list(engine.le.classes_).index(target_label)
            m_df = pd.DataFrame([metrics])
            m_df.columns = m_df.columns.str.strip()
            scaled = result.get('scaled_features')
            if scaled is None:
                scaled = engine.scaler.transform(m_df[engine.features])
            analysis_data = explainer.explain(scaled, metrics, pred_idx)
        except Exception as e:
            logger.error(f"❌ SHAP Analysis Failed: {e}")

        # Send SHAP data
        segment_info = segment_summary or _run_buffer.get(eq_id, {}).get('segment_summary', {})
        segment_item = _segment_top_sensor_to_analysis_item(segment_info.get('top_sensor'))
        analysis_data = _prioritize_representative_root_cause(analysis_data, segment_item)

        if analysis_data:
            shap_payload = {
                "type": "shap_data",
                "equipment_id": eq_id,
                "time": current_time,
                "run_name": run_name,
                "fault_status": fault_status,
                "analysis_data": analysis_data
            }
            await manager.send_json(websocket, shap_payload)

        # --- LLM + GraphRAG 병렬 실행 ---
        # 사용자 요청에 따라 'UNKNOWN FAULT' 대신 '감지된 결함'으로 명칭을 순화하여 리포트 생성
        display_fault_name = fault_status if fault_status != "UNKNOWN FAULT" else "감지된 결함"
        
        explanation, recommendation = await _collect_llm_outputs(
            asyncio.to_thread(_call_shap_agent, display_fault_name, analysis_data),
            asyncio.to_thread(_call_rag_agent, display_fault_name, analysis_data),
        )

        # Send LLM report
        report_payload = {
            "type": "shap_report",
            "equipment_id": eq_id,
            "time": current_time,
            "run_name": run_name,
            "fault_status": fault_status,
            "explanation": explanation,
            "recommendation": recommendation,
            "top_candidates": result.get('top_candidates', []),
            "root_cause_sensor": segment_info.get('top_sensor')
        }
        await manager.send_json(websocket, report_payload)

        # --- Slack Notification ---
        if slack_active:
            try:
                await asyncio.to_thread(
                    _send_slack_alert,
                    fault_status,
                    run_name,
                    result['mse'],
                    result['confidence'],
                    explanation,
                    recommendation,
                    [analysis_data[0].get('sensor')] if analysis_data else [],
                    eq_id,
                    current_time,
                    "analysis",
                )
            except Exception as e:
                logger.error(f"❌ Slack Alert Failed: {e}")

        logger.info(f"✅ Anomaly pipeline completed: {eq_id} | {fault_status} | {run_name}")

    except Exception as e:
        logger.error(f"❌ Background anomaly pipeline crashed for {eq_id}: {e}")


def _call_shap_agent(fault_status: str, analysis_data: list) -> str:
    """Synchronous LLM call (run in thread)."""
    from agents.shap_agent import SHAPAgent
    agent = SHAPAgent()
    return agent.explain_fault(fault_status, analysis_data)


async def _collect_llm_outputs(shap_task, rag_task) -> tuple[str, str]:
    """Return partial LLM outputs even if SHAP or RAG fails independently."""
    shap_result, rag_result = await asyncio.gather(
        shap_task,
        rag_task,
        return_exceptions=True,
    )

    if isinstance(shap_result, Exception):
        logger.error(f"❌ SHAP LLM Failed: {shap_result}")
        explanation = f"Error generating SHAP explanation: {shap_result}"
    else:
        explanation = shap_result

    if isinstance(rag_result, Exception):
        logger.error(f"❌ GraphRAG Failed: {rag_result}")
        recommendation = None
    else:
        recommendation = rag_result

    return explanation, recommendation


def _call_rag_agent(fault_status: str, analysis_data: list) -> str:
    """Synchronous GraphRAG call (run in thread)."""
    from agents.rag_agent import GraphRAGAgent
    agent = GraphRAGAgent()
    try:
        return agent.get_recommendation(fault_status, shap_analysis=analysis_data)
    finally:
        agent.close()


def _send_slack_alert(
    fault_name,
    run_name,
    mse,
    confidence,
    explanation,
    recommendation=None,
    root_cause_sensors=None,
    equipment_id=None,
    event_time=None,
    alert_stage="immediate",
):
    """Synchronous Slack webhook call (run in thread)."""
    if _is_duplicate_slack_alert(equipment_id, event_time, alert_stage):
        logger.info(
            "Skipping duplicate Slack alert for %s at %s (%s)",
            equipment_id or "unknown-equipment",
            event_time or "unknown-time",
            alert_stage,
        )
        return True

    from notifications.slack import SlackNotifier
    notifier = SlackNotifier()
    # Allow temporary override via environment variable SLACK_WEBHOOK_OVERRIDE
    override = os.getenv("SLACK_WEBHOOK_OVERRIDE")
    return notifier.send_alert(
        fault_name,
        run_name,
        mse,
        confidence,
        explanation,
        recommendation=recommendation,
        root_cause_sensors=root_cause_sensors,
        equipment_id=equipment_id,
        event_time=event_time,
        alert_stage=alert_stage,
        webhook_url=override,
    )


def _is_duplicate_slack_alert(equipment_id, event_time, alert_stage) -> bool:
    global _slack_alert_dedupe
    now = time.time()
    cutoff = now - SLACK_DEDUPE_TTL_SECONDS
    _slack_alert_dedupe = {
        key: timestamp for key, timestamp in _slack_alert_dedupe.items()
        if timestamp >= cutoff
    }

    dedupe_key = (
        equipment_id or "unknown-equipment",
        event_time or time.strftime("%H:%M:%S"),
        alert_stage or "alert",
    )
    if dedupe_key in _slack_alert_dedupe:
        return True

    _slack_alert_dedupe[dedupe_key] = now
    return False


def _slack_is_configured() -> bool:
    from notifications.slack import SlackNotifier
    return SlackNotifier().is_configured()


# ==========================================================================
# REST Endpoints
# ==========================================================================
@app.post("/api/rag_search")
async def rag_search(request: RAGSearchRequest):
    """
    GraphRAG 정비 가이드 검색
    
    RagGuide.jsx에서 호출하는 엔드포인트.
    사용자 질의를 Neo4j Knowledge Graph + LLM으로 처리하여 정비 가이드를 반환합니다.
    """
    query = request.query
    fault_name = request.fault_name

    if request.use_v2:
        # GraphRAG Agent V2 (fingerprint matching)
        try:
            result = await asyncio.to_thread(_call_rag_v2, query, fault_name)
            return {
                "recommendation": result.get("answer", "No answer generated"),
                "candidates": result.get("candidates", []),
                "chain": result.get("chain", {}),
                "token_estimate": result.get("token_estimate", 0)
            }
        except Exception as e:
            logger.error(f"RAG V2 search error: {e}")
            return {"recommendation": f"Error: {str(e)}", "candidates": [], "chain": {}}
    else:
        # GraphRAG Agent V1
        try:
            recommendation = await asyncio.to_thread(_call_rag_search_v1, query, fault_name)
            return {"recommendation": recommendation}
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return {"recommendation": f"Error: {str(e)}"}


def _call_rag_search_v1(query: str, fault_name: str = None) -> str:
    """V1 RAG Agent: simple query → recommendation."""
    from agents.rag_agent import GraphRAGAgent
    agent = GraphRAGAgent()
    try:
        # fault_name이 없으면 query에서 추론 시도
        effective_fault = fault_name or query
        return agent.get_recommendation(effective_fault, shap_analysis=None)
    finally:
        agent.close()


def _call_rag_v2(query: str, fault_name: str = None) -> dict:
    """V2 RAG Agent: fingerprint matching + causal chain."""
    from agents.rag_agent import GraphRAGAgentV2
    agent = GraphRAGAgentV2()
    try:
        return agent.recommend(
            fault_name_hint=fault_name or query,
            question=query
        )
    finally:
        agent.close()


@app.get("/api/system_status")
async def system_status():
    """시스템 상태 조회 — 프론트엔드 헤더 바에 표시"""
    uptime = time.time() - startup_time if startup_time else 0

    return {
        "status": "online",
        "uptime_seconds": round(uptime, 1),
        "models": {
            "autoencoder": {
                "type": "Autoencoder (PyTorch)",
                "features": len(engine.features) if engine else 0,
                "threshold": engine.base_threshold if engine else 0
            },
            "classifier": {
                "type": "LightGBM",
                "classes": list(engine.le.classes_) if engine else []
            }
        },
        "pipeline": {
            "kafka_available": KAFKA_AVAILABLE,
            "neo4j_configured": bool(os.getenv("NEO4J_URI")),
            "slack_configured": _slack_is_configured(),
            "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
        },
        "active_connections": len(manager.active_connections)
    }


@app.post("/api/slack_test")
async def slack_test(request: SlackTestRequest):
    """Slack 테스트 알림 발송"""
    try:
        sent = await asyncio.to_thread(
            _send_slack_alert,
            "MANUAL TEST", "TEST_FROM_REACT", 0.0, 1.0,
            request.message,
            "1. 해당 설비를 즉시 확인하고 필요 시 Hold 상태로 전환하세요. 2. 대표 원인 센서의 최근 추세와 공정 조건 변화를 점검하세요. 3. RF/압력/매칭 계통 점검 후 공정 담당자 승인 전까지 재가동을 보류하세요.",
            [
                {"sensor": "RF_Power", "shap_value": 0.4213},
                {"sensor": "Match_Load", "shap_value": -0.2188},
                {"sensor": "Chamber_Pressure", "shap_value": 0.1331},
            ],
            "EQ-TEST",
            time.strftime("%H:%M:%S"),
            "analysis",
        )
        return {
            "success": bool(sent),
            "message": "Test alert sent to Slack" if sent else "Slack is not configured or Slack rejected the message",
        }
    except Exception as e:
        return {"success": False, "message": str(e)}


# ==========================================================================
# Health Check
# ==========================================================================
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


# ==========================================================================
# Entry Point
# ==========================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
