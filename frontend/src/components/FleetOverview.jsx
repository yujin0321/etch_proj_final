import React, { memo } from 'react';
import { Activity, AlertTriangle, CheckCircle, AlertCircle, MoveRight, PauseCircle } from 'lucide-react';

const FleetOverview = ({ fleetStatus, onSelectEquipment }) => {
  const equipments = Array.from({ length: 10 }, (_, i) => `EQ-${String(i + 1).padStart(2, '0')}`);
  const normalCount = equipments.filter((id) => !fleetStatus[id]?.is_anomaly).length;
  const anomalyCount = equipments.filter((id) => fleetStatus[id]?.is_anomaly).length;
  const fleetHealth = ((normalCount / equipments.length) * 100).toFixed(0);

  return (
    <div className="tab-content fleet-overview-page">
      <div className="fleet-overview-heading">
        <h2>전체 설비 현황</h2>
        <span>Etch 챔버 10대 실시간 공정 모니터링</span>
      </div>

      <div className="fleet-summary-grid">
        <div className="fleet-summary-card">
          <div className="summary-copy">
            <p>공정 가동률</p>
            <div className="summary-legend">
              <span><i className="legend-dot cyan" />Fleet Health</span>
            </div>
          </div>
          <strong>{fleetHealth}%</strong>
        </div>

        <div className="fleet-summary-card">
          <div className="summary-copy">
            <p>정상 가동</p>
            <div className="summary-legend">
              <span><CheckCircle size={13} />Active chambers</span>
            </div>
          </div>
          <strong>{normalCount}</strong>
        </div>

        <div className={`fleet-summary-card ${anomalyCount > 0 ? 'warning' : ''}`}>
          <div className="summary-copy">
            <p>이상/정지</p>
            <div className="summary-legend">
              <span><AlertCircle size={13} />Alert chambers</span>
            </div>
          </div>
          <strong>{anomalyCount}</strong>
        </div>
      </div>

      <div className="fleet-equipment-grid">
        {equipments.map((eqId) => {
          const statusData = fleetStatus[eqId] || { mse: 0, status: '대기 중', is_anomaly: false };
          const isAnomaly = Boolean(statusData.is_anomaly);
          const reroutedTo = statusData.reroutedTo;
          const reroutedFrom = statusData.reroutedFrom;

          return (
            <button
              key={eqId}
              type="button"
              onClick={() => onSelectEquipment(eqId)}
              className={`glass-panel equipment-card ${isAnomaly ? 'anomaly-pulse' : ''}`}
            >
              <div className="equipment-card-top">
                <h3>{eqId}</h3>
                {isAnomaly ? <AlertTriangle size={20} /> : <Activity size={20} />}
              </div>

              <div className="equipment-status-line">
                <span className="equipment-label">상태</span>
                <span className={`status-pill ${isAnomaly ? 'alert' : 'normal'}`}>{statusData.status}</span>
              </div>

              <div className="equipment-metric-line">
                <span className="equipment-label">MSE</span>
                <strong className={`equipment-mse ${isAnomaly ? 'alert' : ''}`}>{(statusData.mse || 0).toFixed(4)}</strong>
              </div>

              {(isAnomaly || reroutedFrom) && (
                <div className={`equipment-routing-panel ${isAnomaly ? 'alert' : 'receiving'}`}>
                  {isAnomaly ? (
                    <>
                      <div className="routing-row strong">
                        <PauseCircle size={15} />
                        <span>이상치 발생으로 설비 정지</span>
                      </div>
                      <div className="routing-row">
                        <MoveRight size={15} />
                        <span>잔여 작업: <strong>{reroutedTo || '대기 중'}</strong></span>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="routing-row strong">
                        <MoveRight size={15} />
                        <span>잔여 작업 수신 중</span>
                      </div>
                      <div className="routing-row">
                        <span>이동 원 설비: <strong>{reroutedFrom}</strong></span>
                      </div>
                    </>
                  )}
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default memo(FleetOverview);
