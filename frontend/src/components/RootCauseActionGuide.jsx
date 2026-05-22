import React from 'react';
import { AlertTriangle, Gauge, ShieldCheck, Wrench, Zap } from 'lucide-react';

const SENSOR_LIBRARY = {
  'TCP Top Power': {
    label: 'TCP Top Power',
    area: '상부 RF 전극 / 플라즈마 소스',
    role: '플라즈마 밀도를 만드는 상부 RF 전력 상태를 감시합니다.',
    action: [
      '상부 RF Generator 출력과 Forward/Reflected Power 로그를 먼저 확인합니다.',
      '매칭 네트워크 튜닝 상태와 아킹 이벤트 발생 여부를 점검합니다.',
      '전극 오염 또는 RF 케이블 체결 이상이 있으면 PM 후 더미 웨이퍼로 안정성을 확인합니다.',
    ],
  },
  'Bias Power': {
    label: 'Bias Power',
    area: '웨이퍼 척 / 하부 RF Bias',
    role: '이온 에너지와 식각 프로파일에 영향을 주는 하부 RF 전력을 감시합니다.',
    action: [
      'Bias Generator의 출력 편차와 레시피 세트포인트 일치 여부를 확인합니다.',
      'ESC 접촉 상태, 웨이퍼 클램핑, 하부 매칭 네트워크 반사파를 함께 점검합니다.',
      '반복 편차가 있으면 챔버 클린 후 기준 레시피로 재현성을 검증합니다.',
    ],
  },
  'Chamber Pressure': {
    label: 'Chamber Pressure',
    area: '공정 챔버 / 압력 제어부',
    role: '챔버 내부 압력 안정성과 펌핑 밸런스를 감시합니다.',
    action: [
      '압력 트렌드와 MFC 유량, Throttle/Vat Valve 개도율을 동시에 확인합니다.',
      '압력 드리프트가 크면 누설 점검과 펌프 라인 막힘 여부를 우선 확인합니다.',
      '압력 제어가 복구된 뒤 레시피 시작 전 Base Pressure 도달 시간을 확인합니다.',
    ],
  },
  'He Chuck': {
    label: 'He Chuck',
    area: 'ESC / Backside He 냉각 라인',
    role: '웨이퍼 후면 냉각용 헬륨 압력과 열전달 상태를 감시합니다.',
    action: [
      'Backside He 압력, 누설률, ESC 클램핑 전압을 즉시 확인합니다.',
      '압력 저하가 있으면 웨이퍼 미스얼라인, O-ring, He 라인 누설을 점검합니다.',
      '열 안정성이 확보되기 전까지 고온 민감 레시피 투입을 보류합니다.',
    ],
  },
  'Vat Valve': {
    label: 'Vat Valve',
    area: '배기 라인 / 압력 제어 밸브',
    role: '배기 유량과 챔버 압력 응답성을 제어하는 밸브 상태를 감시합니다.',
    action: [
      '밸브 개도율 명령값과 실제 피드백 값의 차이를 먼저 확인합니다.',
      '스틱션, 구동부 지연, 배기 라인 파티클 누적 여부를 점검합니다.',
      '밸브 보정 후 압력 램프 응답이 기준 시간 내로 들어오는지 확인합니다.',
    ],
  },
  'OES Intensity': {
    label: 'OES Intensity',
    area: 'OES 광학 센서 / 플라즈마 방출 모니터',
    role: '플라즈마 화학종 변화와 엔드포인트 신호를 감시합니다.',
    action: [
      '광학 윈도우 오염, 신호 포화, 기준 파장 강도 변화를 우선 확인합니다.',
      '최근 챔버 클린 이력과 공정 가스 조성 변경 여부를 함께 비교합니다.',
      '윈도우 클리닝 후 기준 웨이퍼로 엔드포인트 신호가 회복되는지 검증합니다.',
    ],
  },
};

const FALLBACK_SENSORS = [
  { name: 'TCP Top Power', value: 0.92 },
  { name: 'Bias Power', value: 0.76 },
  { name: 'Chamber Pressure', value: 0.62 },
  { name: 'He Chuck', value: 0.48 },
  { name: 'Vat Valve', value: 0.36 },
  { name: 'OES Intensity', value: 0.28 },
];

const cleanSensorName = (name = '') => String(name || '')
  .split('__', 1)[0]
  .replace(/_/g, ' ')
  .trim();

const NON_SENSOR_NAMES = new Set([
  'run name',
  'run id',
  'fault name',
  'fault',
  'status',
  'time',
  'time step',
  'timestamp',
  'equipment id',
  'recipe',
  'label',
  'predicted label',
]);

const isSensorName = (name = '') => {
  const cleaned = cleanSensorName(name).toLowerCase();
  if (!cleaned || NON_SENSOR_NAMES.has(cleaned)) return false;
  if (cleaned.endsWith(' id') || cleaned.endsWith(' name')) return false;
  return true;
};

const normalizeSensorName = (name = '') => {
  const cleaned = cleanSensorName(name);
  const lower = cleaned.toLowerCase();
  if (lower.includes('tcp')) return 'TCP Top Power';
  if (lower.includes('bias')) return 'Bias Power';
  if (lower.includes('pressure') || lower.includes('chamber')) return 'Chamber Pressure';
  if (lower.includes('he') || lower.includes('chuck')) return 'He Chuck';
  if (lower.includes('vat') || lower.includes('valve')) return 'Vat Valve';
  if (lower.includes('oes')) return 'OES Intensity';
  return cleaned || 'Unknown Sensor';
};

const getSensorInfo = (name) => {
  const normalized = normalizeSensorName(name);
  return SENSOR_LIBRARY[normalized] || {
    label: normalized,
    area: '공정 센서 / 장비 상태 계측부',
    role: '공정 이상 판단에 기여한 주요 센서입니다.',
    action: [
      '해당 센서의 실시간 값과 최근 30분 트렌드를 기준값과 비교합니다.',
      '센서 캘리브레이션, 통신 상태, 레시피 세트포인트 변경 이력을 확인합니다.',
      '동일 증상이 반복되면 장비 엔지니어에게 센서 계측 계통 점검을 요청합니다.',
    ],
  };
};

const RootCauseActionGuide = ({ shapData, topCandidates, rootCauseSensor, selectedEquipment, isRunning }) => {
  const rawPrimaryRootCause = cleanSensorName(rootCauseSensor?.sensor || rootCauseSensor || '');
  const primaryRootCause = isSensorName(rawPrimaryRootCause) ? rawPrimaryRootCause : '';
  const sensorList = ((shapData && shapData.length > 0 ? shapData : FALLBACK_SENSORS)
    .filter((sensor) => isSensorName(sensor.name))
    .slice(0, 6)
    .map((sensor) => ({
      ...sensor,
      name: normalizeSensorName(sensor.name),
      value: Number(sensor.value || 0),
    })));
  const displaySensors = sensorList.length ? sensorList : FALLBACK_SENSORS;
  const sourceSensors = primaryRootCause
    ? displaySensors.sort((a, b) => {
        if (a.name === normalizeSensorName(primaryRootCause)) return -1;
        if (b.name === normalizeSensorName(primaryRootCause)) return 1;
        return b.value - a.value;
      })
    : displaySensors;

  const primarySensor = sourceSensors[0];
  const primaryInfo = getSensorInfo(primarySensor?.name);
  const statusText = isRunning ? '실시간 분석 중' : '대기 상태';

  const orbitPositions = [
    { left: '50%', top: '16%' },
    { left: '22%', top: '39%' },
    { left: '78%', top: '39%' },
    { left: '30%', top: '76%' },
    { left: '70%', top: '76%' },
    { left: '50%', top: '88%' },
  ];

  return (
    <div className="tab-content root-cause-page">
      <section className="root-cause-hero">
        <div>
          <p className="section-kicker">Root Cause & Maintenance Guide</p>
          <h2>이상 원인·조치 가이드</h2>
          <p>
            이상 발생에 기여한 센서를 원형 맵으로 배치하고, 장비 위치와 즉시 조치 순서를 함께 제공합니다.
          </p>
        </div>
        <div className={`status-pill ${isRunning ? 'normal' : 'standby'}`}>{statusText}</div>
      </section>

      <div className="root-cause-layout">
        <div className="root-cause-left">
          <section className="glass-panel sensor-map-panel">
            <div className="panel-heading">
              <div>
                <p className="section-kicker">Selected Equipment</p>
                <h3>{selectedEquipment} 결함 진단 핵심 근거 센서</h3>
              </div>
              <span className="confidence-pill">
                <AlertTriangle size={15} />
                Top 원인: {primaryInfo.label}
              </span>
            </div>
            {primaryRootCause && (
              <div className="root-cause-callout" style={{ marginTop: '1rem', padding: '1rem', borderRadius: '16px', background: 'rgba(255, 239, 213, 0.16)', border: '1px solid rgba(255, 193, 7, 0.14)' }}>
                <strong style={{ display: 'block', marginBottom: '0.5rem' }}>이상 구간 주요 센서</strong>
                <p style={{ margin: 0, color: 'var(--text-primary)' }}>{primaryRootCause}</p>
              </div>
            )}
            <div className="sensor-orbit" aria-label="이상 원인 센서 원형 그래프">
              <div className="orbit-ring orbit-ring-outer" />
              <div className="orbit-ring orbit-ring-inner" />
              <div className="orbit-center">
                <Zap size={24} />
                <strong>{selectedEquipment}</strong>
                <span>이상 집중</span>
              </div>
              <div className="sensor-primary-focus">
                <span>대표 이상 센서</span>
                <strong>{primaryInfo.label}</strong>
              </div>
              {sourceSensors.map((sensor, index) => (
                <div
                  key={`${sensor.name}-${index}`}
                  className={'sensor-node ' + (index === 0 ? 'primary' : '')}
                  style={orbitPositions[index] || orbitPositions[orbitPositions.length - 1]}
                >
                  <strong>{index + 1}</strong>
                  <span>{sensor.name}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="sensor-description-grid">
            {sourceSensors.map((sensor, index) => {
              const info = getSensorInfo(sensor.name);
              return (
                <article key={`${sensor.name}-${index}`} className="sensor-description-card">
                  <div className="sensor-card-icon">
                    <Gauge size={18} />
                  </div>
                  <div>
                    <h4>{info.label}</h4>
                    <p className="sensor-area">{info.area}</p>
                    <p>{info.role}</p>
                  </div>
                </article>
              );
            })}
          </section>
        </div>

        <aside className="glass-panel action-guide-panel">
          <div className="panel-heading">
            <div>
              <p className="section-kicker">Manual & Action</p>
              <h3>우선 조치 가이드</h3>
            </div>
            <Wrench size={24} color="var(--accent-cyan)" />
          </div>

          <div className="priority-callout">
            <ShieldCheck size={20} />
            <div>
              <strong>{primaryInfo.label} 신호가 가장 먼저 흔들렸습니다.</strong>
              <p>{primaryInfo.area}를 기준으로 원인 분리 후 재가동 조건을 확인하세요.</p>
            </div>
          </div>

            <div className="action-section">
            <h4>1. 즉시 확인</h4>
            <ol>
              {primaryInfo.action.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ol>
          </div>

          <div className="action-section">
            <h4>2. 연계 점검</h4>
            <ol>
              {sourceSensors.slice(1, 4).map((sensor) => {
                const info = getSensorInfo(sensor.name);
                return (
                  <li key={sensor.name}>
                    <strong>{info.label}</strong>: {info.area}의 트렌드를 함께 비교해 단일 센서 이상인지 공정 조건 변화인지 구분합니다.
                  </li>
                );
              })}
            </ol>
          </div>

          <div className="action-section">
            <h4>3. 재가동 기준</h4>
            <ol>
              <li>주요 센서 3개 이상의 값이 기준 범위로 복귀했는지 확인합니다.</li>
              <li>알람 발생 시점 이후 동일 레시피 더미 웨이퍼에서 MSE가 안정화되는지 검증합니다.</li>
              <li>반복 알람이 있으면 자동 재가동을 보류하고 PM 또는 부품 점검 이력으로 escalate합니다.</li>
            </ol>
          </div>

          {topCandidates && topCandidates.length > 0 && (
            <div className="candidate-stack">
              <h4>AI 후보 원인</h4>
              {topCandidates.slice(0, 3).map((candidate, index) => (
                <div key={`${candidate.label}-${index}`} className="candidate-row">
                  <span>{index + 1}. {candidate.label}</span>
                  <strong>{Math.round((candidate.confidence || 0) * 100)}%</strong>
                </div>
              ))}
            </div>
          )}
        </aside>
      </div>
    </div>
  );
};

export default RootCauseActionGuide;
