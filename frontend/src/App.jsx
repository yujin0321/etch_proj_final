import React, { useEffect, useRef, useState } from 'react';
import { Activity, BrainCircuit, LayoutDashboard, Play, Square } from 'lucide-react';
import './App.css';
import LiveDashboard from './LiveDashboard';
import RootCauseActionGuide from './components/RootCauseActionGuide';
import FleetOverview from './components/FleetOverview';

function App() {
  const [activeTab, setActiveTab] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [selectedModel] = useState('식각 공정 이상 탐지 모델 v1.0');
  const [fleetStatus, setFleetStatus] = useState({});
  const [metricsHistory, setMetricsHistory] = useState({});
  const [shapHistory, setShapHistory] = useState({});
  const [selectedEquipment, setSelectedEquipment] = useState('EQ-01');
  const [selectedShapTime, setSelectedShapTime] = useState(null);
  const [systemStatus, setSystemStatus] = useState(null);
  const wsRef = useRef(null);

  const dataSource = 'local';
  const slackEnabled = systemStatus?.pipeline?.slack_configured ?? true;
  const streamSpeed = 0.5;

  const handleSelectEquipment = (eqId) => {
    setSelectedEquipment(eqId);
    setActiveTab(1);
  };

  const handleAnomalyClick = (time) => {
    setSelectedShapTime(time);
    setActiveTab(2);
  };

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch('/api/system_status');
        if (res.ok) {
          const data = await res.json();
          setSystemStatus(data);
        }
      } catch (err) {
        console.log('Server not available yet:', err.message);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!isRunning) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      return;
    }

    setMetricsHistory({});
    setShapHistory({});
    setFleetStatus({});

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const params = new URLSearchParams({
      source: dataSource,
      speed: streamSpeed.toString(),
      slack: slackEnabled.toString(),
    });
    const socket = new WebSocket(`${wsProtocol}//${window.location.host}/ws/stream?${params.toString()}`);
    wsRef.current = socket;

    socket.onopen = () => console.log('WebSocket connected');
    socket.onclose = (event) => console.log(`WebSocket closed: ${event.code} ${event.reason}`);
    socket.onerror = (event) => console.error('WebSocket error:', event);

    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        const eqId = payload.equipment_id;
        if (!eqId) return;

        if (payload.type === 'metrics') {
          setFleetStatus((prev) => ({
            ...prev,
            [eqId]: {
              ...(prev[eqId] || {}),
              ...payload,
              status: prev[eqId]?.reroutedFrom ? '잔여 작업 수신 중' : payload.status,
              reroutedFrom: prev[eqId]?.reroutedFrom,
              reroutedTo: prev[eqId]?.reroutedTo,
              isStopped: prev[eqId]?.isStopped,
              is_anomaly: prev[eqId]?.isStopped ? true : payload.is_anomaly,
            },
          }));

          setMetricsHistory((prev) => {
            const eqHistory = prev[eqId] || [];
            const newData = [
              ...eqHistory,
              {
                time: payload.time,
                mse: payload.mse,
                is_anomaly: payload.is_anomaly,
              },
            ];

            return {
              ...prev,
              [eqId]: newData.length > 200 ? newData.slice(1) : newData,
            };
          });
        } else if (payload.type === 'shap_data') {
          const formatted = payload.analysis_data
            .map((item) => ({
              name: item.sensor,
              value: Math.abs(item.shap_value),
            }))
            .sort((a, b) => b.value - a.value)
            .slice(0, 8);

          setShapHistory((prev) => {
            const eqShap = prev[eqId] || {};
            const newEqShap = {
              ...eqShap,
              [payload.time]: {
                data: formatted,
                explanation: 'Analyzing...',
                recommendation: '',
                fault_status: payload.fault_status,
              },
            };

            const times = Object.keys(newEqShap).sort();
            if (times.length > 50) {
              delete newEqShap[times[0]];
            }

            return {
              ...prev,
              [eqId]: newEqShap,
            };
          });

          setSelectedShapTime(payload.time);
        } else if (payload.type === 'shap_report') {
          setShapHistory((prev) => {
            const eqShap = prev[eqId] || {};
            if (!eqShap[payload.time]) return prev;

            return {
              ...prev,
              [eqId]: {
                ...eqShap,
                [payload.time]: {
                  ...eqShap[payload.time],
                  explanation: payload.explanation,
                  recommendation: payload.recommendation,
                  top_candidates: payload.top_candidates || [],
                  root_cause_sensor: payload.root_cause_sensor || null,
                },
              },
            };
          });

          setFleetStatus((prev) => ({
            ...prev,
            [eqId]: {
              ...(prev[eqId] || {}),
              root_cause_sensor: payload.root_cause_sensor || null,
            },
          }));
        } else if (payload.type === 'equipment_stop') {
          setFleetStatus((prev) => ({
            ...prev,
            [eqId]: {
              ...prev[eqId],
              status: '이상치 발생으로 설비 정지',
              is_anomaly: true,
              isStopped: true,
              message: payload.message,
            },
          }));
        } else if (payload.type === 'equipment_reroute') {
          setFleetStatus((prev) => ({
            ...prev,
            [payload.equipment_id]: {
              ...prev[payload.equipment_id],
              status: '설비 이상 발생',
              is_anomaly: true,
              isStopped: true,
              reroutedTo: payload.target_equipment_id,
              message: payload.message,
            },
            [payload.target_equipment_id]: {
              ...(prev[payload.target_equipment_id] || {}),
              status: '잔여 작업 수신 중',
              reroutedFrom: payload.equipment_id,
              is_anomaly: false,
            },
          }));
        } else if (payload.type === 'alert') {
          console.log(`Phase 1 Alert: ${payload.message}`);
          setFleetStatus((prev) => ({
            ...prev,
            [eqId]: {
              ...prev[eqId],
              status: prev[eqId]?.isStopped ? '이상치 발생으로 설비 정지' : '이상 원인 분석 중',
              is_anomaly: true,
            },
          }));
        } else if (payload.type === 'info') {
          console.info('System info:', payload.message);
        } else if (payload.type === 'error') {
          console.error('Server error:', payload.message);
        }
      } catch (err) {
        console.error('Failed to parse WS message:', err);
      }
    };

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [isRunning, slackEnabled]);

  const tabs = [
    { id: 0, label: '전체 설비 현황', icon: LayoutDashboard },
    { id: 1, label: '설비 모니터링', icon: Activity },
    { id: 2, label: '원인·조치 가이드', icon: BrainCircuit },
  ];

  const eqShapHistory = shapHistory[selectedEquipment] || {};
  const shapTimes = Object.keys(eqShapHistory).sort();
  const latestShapTime = shapTimes.length ? shapTimes[shapTimes.length - 1] : null;
  const currentShap = (selectedShapTime && eqShapHistory[selectedShapTime]) ||
    (latestShapTime && eqShapHistory[latestShapTime]) || {
      data: [],
      explanation: '',
      top_candidates: [],
      root_cause_sensor: null,
    };

  return (
    <div className="app-container" style={{ flexDirection: 'column' }}>
      <header
        style={{
          background: 'var(--bg-card)',
          borderBottom: '1px solid var(--border-color)',
          zIndex: 10,
        }}
      >
        <div className="dashboard-header-top">
          <div className="dashboard-title-block">
            <h1>SMART FACTORY</h1>
            <p>식각 공정 이상 탐지 및 AI 원인 분석 대시보드</p>
          </div>

          <div className="dashboard-controls">
            <button className={`start-btn ${isRunning ? 'running' : ''}`} onClick={() => setIsRunning(!isRunning)}>
              {isRunning ? (
                <>
                  <Square size={18} /> 시스템 중지
                </>
              ) : (
                <>
                  <Play size={18} fill="currentColor" /> 시스템 시작
                </>
              )}
            </button>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '1rem', padding: '0 2rem' }}>
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <div
                key={tab.id}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  padding: '0.8rem 1.5rem',
                  cursor: 'pointer',
                  color: activeTab === tab.id ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                  borderBottom: activeTab === tab.id ? '3px solid var(--accent-cyan)' : '3px solid transparent',
                  fontWeight: activeTab === tab.id ? 'bold' : 'normal',
                  transition: 'all 0.2s',
                  fontSize: '1.05rem',
                }}
                onClick={() => setActiveTab(tab.id)}
              >
                <Icon size={20} />
                {tab.label}
              </div>
            );
          })}

          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '1rem', paddingRight: '1rem' }}>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
              선택 설비:{' '}
              <strong style={{ color: 'var(--accent-cyan)', fontSize: '1.1rem' }}>{selectedEquipment}</strong>
            </span>
            {systemStatus && (
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
                | 서버: <span style={{ color: 'var(--accent-green)' }}>{systemStatus.status === 'online' ? '온라인' : systemStatus.status}</span>
                {' | '}
                연결: {systemStatus.active_connections}
              </span>
            )}
          </div>
        </div>
      </header>

      <main className="main-content" style={{ flex: 1, overflowY: 'auto', padding: '0' }}>
        {activeTab === 0 && <FleetOverview fleetStatus={fleetStatus} onSelectEquipment={handleSelectEquipment} />}

        {activeTab === 1 && (
          <LiveDashboard
            latestMetrics={fleetStatus[selectedEquipment] || { mse: 0, status: 'Waiting', confidence: 0, is_anomaly: false }}
            metricsHistory={metricsHistory[selectedEquipment] || []}
            isRunning={isRunning}
            selectedModel={selectedModel}
            onAnomalyClick={handleAnomalyClick}
            equipmentId={selectedEquipment}
            topCandidates={fleetStatus[selectedEquipment]?.top_candidates || []}
            rootCauseSensor={fleetStatus[selectedEquipment]?.root_cause_sensor}
          />
        )}

        {activeTab === 2 && (
          <RootCauseActionGuide
            shapData={currentShap.data}
            topCandidates={currentShap.top_candidates}
            rootCauseSensor={currentShap.root_cause_sensor}
            isRunning={isRunning}
            selectedEquipment={selectedEquipment}
          />
        )}
      </main>
    </div>
  );
}

export default App;
