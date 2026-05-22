import React, { memo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Brush } from 'recharts';
import { Activity, AlertTriangle } from 'lucide-react';

const CustomDot = (props) => {
  const { cx, cy, payload, onAnomalyClick } = props;

  if (payload.is_anomaly) {
    return (
      <circle
        cx={cx}
        cy={cy}
        r={6}
        fill="var(--accent-red)"
        stroke="white"
        strokeWidth={1.5}
        style={{ cursor: 'pointer' }}
        onClick={() => onAnomalyClick && onAnomalyClick(payload.time)}
      >
        <title>Open root-cause analysis</title>
      </circle>
    );
  }

  return null;
};

const formatStatus = (status, isRunning, isAnomaly) => {
  if (!isRunning) return '대기 중';
  if (isAnomaly) return status && status !== 'Normal' && status !== 'Waiting'
    ? status
    : '이상 감지';
  if (status === 'Waiting') return '대기 중';
  if (status === 'Normal') return '정상 운전';
  return status || '정상 운전';
};

const LiveDashboard = ({ latestMetrics, metricsHistory, isRunning, selectedModel, onAnomalyClick, equipmentId, topCandidates, rootCauseSensor }) => {
  const { mse = 0, status = 'Waiting', confidence = 0, is_anomaly = false } = latestMetrics;
  const rootCauseName = rootCauseSensor?.sensor || rootCauseSensor;
  const displayStatus = formatStatus(status, isRunning, is_anomaly);

  return (
    <div className="tab-content">
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
        <div className="glass-panel">
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>설비 ID</p>
          <h3 style={{ fontSize: '1.5rem', marginTop: '0.5rem', color: 'var(--accent-cyan)' }}>{equipmentId || 'EQ-01'} (Etch)</h3>
        </div>
        <div className="glass-panel">
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>현재 이상 점수 (MSE)</p>
          <h3 style={{ fontSize: '1.5rem', marginTop: '0.5rem', color: is_anomaly ? 'var(--accent-red)' : 'white' }}>
            {mse.toFixed(4)}
          </h3>
        </div>
        <div className="glass-panel">
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>결함 분류 신뢰도</p>
          <h3 style={{ fontSize: '1.5rem', marginTop: '0.5rem' }}>{(confidence * 100).toFixed(1)}%</h3>
        </div>
        <div className={`glass-panel ${is_anomaly ? 'anomaly-pulse' : ''}`}>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>현재 상태</p>
          <h3 style={{ fontSize: '1.5rem', marginTop: '0.5rem', color: is_anomaly ? 'var(--accent-red)' : 'var(--accent-green)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            {is_anomaly ? <AlertTriangle size={24} /> : <Activity size={24} />}
            {displayStatus}
          </h3>
        </div>

        {is_anomaly && rootCauseName && (
          <div className="glass-panel" style={{ gridColumn: 'span 4', background: 'rgba(255, 228, 181, 0.18)', animation: 'fadeIn 0.5s ease' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <p style={{ color: 'var(--accent-orange)', fontSize: '0.9rem', fontWeight: 'bold', whiteSpace: 'nowrap' }}>이상 구간 대표 원인 센서</p>
              <div style={{ height: '1px', flex: 1, background: 'rgba(248, 113, 113, 0.18)' }}></div>
              <div style={{ padding: '0.3rem 0.8rem', borderRadius: '9999px', background: 'rgba(255, 165, 0, 0.14)' }}>
                <span style={{ color: 'var(--text-primary)', fontWeight: 'bold' }}>{rootCauseName}</span>
              </div>
            </div>
          </div>
        )}

        {is_anomaly && topCandidates && topCandidates.length > 0 && (
          <div className="glass-panel" style={{ gridColumn: 'span 4', background: 'rgba(248, 113, 113, 0.08)', animation: 'fadeIn 0.5s ease' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <p style={{ color: 'var(--accent-red)', fontSize: '0.9rem', fontWeight: 'bold', whiteSpace: 'nowrap' }}>추정 결함 원인 TOP 3</p>
              <div style={{ height: '1px', flex: 1, background: 'rgba(248, 113, 113, 0.18)' }}></div>
              <div style={{ display: 'flex', gap: '2rem' }}>
                {topCandidates.map((candidate, index) => (
                  <div key={`${candidate.label}-${index}`} style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', padding: '0.3rem 0.8rem', borderRadius: '9999px', background: 'rgba(160, 174, 192, 0.08)' }}>
                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>{index + 1}</span>
                    <span style={{ color: 'var(--text-primary)', fontWeight: 'bold' }}>{candidate.label}</span>
                    <span style={{ color: 'var(--accent-cyan)', fontSize: '0.85rem' }}>{(candidate.confidence * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="glass-panel" style={{ height: '350px' }}>
        <h3 style={{ marginBottom: '1rem', fontSize: '1rem', color: 'var(--text-secondary)' }}>
          실시간 이상 점수 (MSE)
        </h3>
        <div style={{ height: 'calc(100% - 30px)', width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={metricsHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(160, 174, 192, 0.14)" />
              <XAxis dataKey="time" stroke="var(--text-secondary)" />
              <YAxis domain={['auto', 'auto']} stroke="var(--text-secondary)" />
              <Tooltip contentStyle={{ backgroundColor: '#262730', border: '0', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.3)' }} />
              <Line
                type="monotone"
                dataKey="mse"
                stroke="var(--accent-blue)"
                strokeWidth={2}
                dot={<CustomDot onAnomalyClick={onAnomalyClick} />}
                activeDot={{ r: 8 }}
                isAnimationActive={false}
              />
              <Brush dataKey="time" height={30} stroke="var(--border-color)" fill="var(--bg-card)" tickFormatter={() => ''} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="glass-panel">
        <h3 style={{ marginBottom: '1rem', fontSize: '1rem' }}>시스템 이벤트 로그</h3>
        <table className="logs-table">
          <thead>
            <tr>
              <th>시간</th>
              <th>이벤트</th>
              <th>모델</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>{new Date().toLocaleTimeString()}</td>
              <td style={{ color: is_anomaly && isRunning ? 'var(--accent-red)' : 'inherit' }}>
                {displayStatus}
              </td>
              <td>{selectedModel}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default memo(LiveDashboard);
