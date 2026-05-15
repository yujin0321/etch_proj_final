"""
Engineer-facing diagnostic-report prompts for the 20 plasma-etch fault types.

장비 가정: TCP/ICP 플라즈마 에처 (LAM 9600 계열),
공정 화학: BCl3 + Cl2 (+ He cooling) 기반 Al/poly-Si/metal etching.

[대응 절차 출처 — 공개 자료 기반]
- IBM / Semiconductor Digest (2010): Gas Flow Monitoring 4단계 표준 (Alert → Halt
  → Recalibrate → Feedback control). MFC out-of-spec 대응의 fab 표준 워크플로.
- Advanced Energy 백서 / Coaxial Power Systems FAQ: RF Reflected Power > 20% rated
  발생 시 점검 순서 — generator → cable/coax → match network → load 순으로 isolation.
- MDPI Electronics 10(17), 2074 (2021): Dual-frequency RF matching impedance drift
  현상과 reflected power 패턴 분석.
- CSMantech / Foamtec WCC / MIT Labnetwork archives: ESC backside He leak — He flow
  ≥ 20 sccm = complete clamp failure 기준, O-ring 교체 및 클램프 테스트 절차.
- USPTO patents (US7181306, US7067432 등): throttle valve 기반 chamber pressure
  control 및 chamber drift 모니터링 표준 sequence.
- Lieberman & Lichtenberg, "Principles of Plasma Discharges" (텍스트북): plasma
  parameter ↔ etch outcome 인과 관계의 일반론.

위 자료는 모두 공개 접근 가능하며, 본 prompt의 대응 절차는 사내 매뉴얼이 아닌
일반적·교과서적 fab 관행을 따릅니다. 실제 production 적용 시에는 해당 fab의
SOP를 우선해야 합니다.

[사용 흐름]
    1) 분류기가 fault 라벨을 결정
    2) 같은 run의 센서 mean/std 통계를 dict로 정리
    3) build_prompt(fault_label, stats) 호출 → (system, user) 문자열 쌍 반환
    4) LLM API에 messages=[{"role":"system",...},{"role":"user",...}] 형태로 전송
"""

# ---------------------------------------------------------------------------
# 1) 공통 SYSTEM PROMPT
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
당신은 반도체 TCP/ICP 플라즈마 에처의 FDC(Fault Detection & Classification) 진단 보조 시스템입니다. \
대상 공정은 BCl3 + Cl2 기반의 Al/poly-Si/metal etching이며, 입력으로 다음을 받습니다.

[입력 구성]
1. 분류 결과 (Classified Fault Label)
2. 센서 통계 요약 (mean ± std, 한 run 기준)
   - MACHINE : 레시피 파라미터 (BCl3 Flow, Cl2 Flow, Pressure, He Press, RF Pwr, TCP Top Pwr 등)
   - OES     : Optical Emission Spectroscopy, 250–791 nm 파장별 강도
   - RFM     : RF Metrology, S1–S34 harmonics의 voltage/current
3. Fault Knowledge Card (라벨에 해당하는 사전 지식 — 물리적 의미, 예상 시그니처, 근본 원인 후보, 일반적 대응 절차)

[중요한 면책 조항]
- Knowledge Card의 대응 절차는 IBM/Advanced Energy/학술 논문/USPTO 등 공개 자료에서 \
정리한 fab 일반 관행이며, 특정 회사의 SOP가 아닙니다. \
LLM은 출력 시 "참고용 일반 절차이며 실제 적용 전 자사 SOP 우선" 임을 항상 명시.

[출력 규칙]
- 한국어로 작성하되 장비/공정 전문 용어는 영문 그대로 사용 (예: matching network, throttle valve, anisotropy).
- 입력에 없는 수치는 절대 만들어내지 말 것. Knowledge Card는 prior일 뿐, 실제 통계가 우선.
- 라벨과 관측 시그니처가 모순되면 명시적으로 flag.
- 대응 절차는 Knowledge Card의 "response" 필드를 기반으로 하되, 관측된 통계에 맞게 우선순위 조정.

[출력 형식 — 반드시 이 구조로 작성]

【결함 요약】
- 분류 결과: <label>
- 물리적 의미: <한 줄>
- 심각도: Low / Medium / High / Critical — <사유 한 줄>

【관측된 시그니처】
- MACHINE: <어느 채널이 nominal 대비 얼마나 벗어났는지, 통계값 인용>
- OES    : <비정상 파장과 의미하는 chemical species 변화>
- RFM    : <impedance/load 변화, harmonics 분포, plasma 상태 함의>
- 라벨 vs. 관측 일치도: 일치 / 부분 일치 / 불일치 (불일치면 이유 명시)

【추정 근본 원인】 (가능성 순으로 나열)
1. <원인> — <근거>
2. <원인> — <근거>
3. <원인> — <근거>

【공정 영향】
- Etch rate / Selectivity / CD uniformity / Profile에 대한 예상 영향
- Wafer 폐기 위험도

【대응 절차】 (Knowledge Card의 일반 절차 기반, 자사 SOP 우선 적용 필요)
- 즉시 조치 (현재 run/lot 내):
  1. <구체적 step>
  2. <구체적 step>
- 단기 조치 (수 시간~하루 내):
  1. <구체적 step>
  2. <구체적 step>
- 재가동 전 검증 항목:
  1. <확인 항목>
  2. <확인 항목>
- 장기 예방 (PM cycle / SPC 관점):
  1. <예방 조치>

【주의】
- 본 대응 절차는 공개 자료(IBM/Advanced Energy/학술 논문 등) 기반 일반 관행입니다.
- 실제 적용 전 반드시 자사 SOP 및 장비 매뉴얼 우선 확인 필요.
"""


# ---------------------------------------------------------------------------
# 2) USER PROMPT 템플릿
# ---------------------------------------------------------------------------
USER_TEMPLATE = """\
[Classified Fault]
{fault_label}

[Sensor Statistics — mean ± std (run-level)]
{stats_block}

[Fault Knowledge Card — prior for "{fault_label}"]

▣ 물리적 의미       : {physical}
▣ 예상 MACHINE 시그니처: {machine}
▣ 예상 OES 시그니처    : {oes}
▣ 예상 RFM 시그니처    : {rfm}
▣ 근본 원인 후보 (가능성 순):
{causes_block}
▣ 일반적 공정 영향     : {impact}
▣ Default 심각도       : {severity}

▣ 대응 절차 (공개 자료 기반, 자사 SOP 우선):
  ◦ 즉시 (현재 run/lot 내):
{resp_immediate}
  ◦ 단기 (수 시간~하루 내):
{resp_short}
  ◦ 재가동 전 검증:
{resp_verify}
  ◦ 장기 예방:
{resp_prevent}

[Task]
위 system prompt의 출력 형식대로 진단 리포트를 작성하시오.
- 입력된 센서 통계와 Knowledge Card prior를 대조하여 일치/모순 평가.
- 대응 절차는 Knowledge Card 기반이되 관측된 통계로 우선순위 재조정.
- 모든 대응 항목 끝에 "(자사 SOP 우선 확인)" 명기.
"""


# ---------------------------------------------------------------------------
# 3) 20개 결함별 KNOWLEDGE CARD (대응 절차 포함)
# ---------------------------------------------------------------------------
# 대응 절차의 구조:
#   immediate  : 현재 run/lot 손실 최소화를 위한 즉각 조치
#   short_term : 결함 원인 격리·복구 작업
#   verify     : production 재개 전 확인 항목 (chamber seasoning 등)
#   prevent    : SPC 한계·PM 주기 조정 같은 재발 방지

FAULT_KNOWLEDGE = {

    # ═══════════════════════ BCl3 계열 ═══════════════════════
    "BCl3 +5": {
        "physical": "BCl3 MFC setpoint이 nominal 대비 +5 sccm 초과. 챔버 내 BCl3 부분압 가볍게 상승.",
        "machine":  "BCl3 Flow 채널 평균이 nominal 대비 약 +5 sccm. Cl2/He/TCP/RF는 nominal 유지.",
        "oes":      "BCl/BCl2 band(250–278 nm) emission ↑, AlCl(284 nm, Al-etch product) 동반 ↑.",
        "rfm":      "가스 혼합비 변화로 plasma impedance |Z| 소폭 ↑. harmonics 분포 거의 유지.",
        "causes":   [
            "BCl3 MFC drift / zero offset (가장 흔함)",
            "MFC controller calibration aging",
            "BCl3 upstream pressure regulator 미세 drift",
            "Pneumatic valve seat의 미세 누설",
        ],
        "impact":   "BT(breakthrough) step etch rate 약간 ↑, polymer 형성 균형 변화. 일반적으로 wafer 폐기 수준은 아님.",
        "severity": "Low–Medium",
        "response": {
            "immediate": [
                "현재 wafer는 in-spec 가능성 높으나 lot 단위 SPC 결과 확인 후 hold/release 결정",
                "BCl3 MFC reading과 actual flow의 괴리 여부를 GFM(Gas Flow Monitor) 데이터로 확인",
                "후속 lot은 SPC 한계 close-watch 모드 (IBM/Semiconductor Digest 워크플로)",
            ],
            "short_term": [
                "Zero-flow check 실행: BCl3 line valve 닫고 MFC 0 sccm 명령, reading이 FS의 1% 이내인지 확인",
                "Zero offset 발견 시 MFC 벤더 절차대로 zero calibration",
                "Setpoint flow check: 50% / 100% FS 두 지점에서 verified flow와 비교",
            ],
            "verify": [
                "Zero/span calibration 후 monitor wafer 1매 etch → etch rate가 SPC 중심값 ±2σ 내",
                "OES BCl band(272.2 nm) intensity가 nominal envelope 복귀",
            ],
            "prevent": [
                "BCl3 MFC zero calibration 주기를 짧게 조정 (예: 분기 → 월)",
                "SPC chart에 BCl3 Flow control limit 조정 검토",
            ],
        },
    },

    "BCl3 +10": {
        "physical": "BCl3 MFC setpoint이 nominal 대비 +10 sccm 초과 (BCl3 +5의 약 2배 excursion).",
        "machine":  "BCl3 Flow가 nominal +~10 sccm. MFC reading과 실제 flow의 괴리 가능성 확인 필요.",
        "oes":      "BCl band emission이 +5 대비 뚜렷이 ↑. AlCl 상승 폭 약 2배.",
        "rfm":      "Impedance shift 명확. matching tune이 nominal envelope을 벗어날 수 있음.",
        "causes":   [
            "BCl3 MFC control loop 이탈 (full-scale offset 또는 valve stuck-open)",
            "MFC firmware/communication issue로 actual ≠ reported",
            "Upstream pressure regulator 고장으로 line pressure 상승",
            "MFC zero recalibration 장기 미수행",
        ],
        "impact":   "BT etch rate 뚜렷이 ↑. CD bias 증가 가능. 후속 ME step profile에 영향.",
        "severity": "Medium",
        "response": {
            "immediate": [
                "Wafer 처리 정지 + 현재 lot hold (IBM 워크플로 Step 2: Halt)",
                "EQ engineer 호출, BCl3 line 격리 valve 차단 검토",
                "Hold된 wafer는 cross-section 또는 inline metrology로 CD/profile 확인 후 폐기·재가공 결정",
            ],
            "short_term": [
                "BCl3 MFC 교체 또는 in-place recalibration (벤더 절차)",
                "MFC 교체 시 controller firmware 버전과 ID 정합 확인",
                "Upstream regulator 출력압 측정 → 스펙(보통 30 psig 부근) 대비 검증",
            ],
            "verify": [
                "교체/calibration 후 dry-run (no wafer) 5회 → BCl3 Flow std가 nominal 수준 복귀",
                "Monitor wafer 1–3매 etch → etch rate, CD uniformity SPC 내 회귀",
                "Chamber seasoning 1–2시간 (chamber wall condition 안정화)",
            ],
            "prevent": [
                "BCl3 MFC SPC limit 재검토 (±2σ → ±1.5σ tightening 고려)",
                "MFC age tracking → 일정 사용 시간 초과 시 proactive 교체 PM",
            ],
        },
    },

    "BCl3 -5": {
        "physical": "BCl3 setpoint이 nominal 대비 −5 sccm 부족.",
        "machine":  "BCl3 Flow가 nominal 대비 약 5 sccm 낮음.",
        "oes":      "BCl/BCl2 band(250–278 nm) ↓. AlCl(284 nm) 약화. native oxide 침투 단계 emission signature 약화.",
        "rfm":      "Impedance 소폭 변화. dominant species(Cl) 변화는 미미.",
        "causes":   [
            "BCl3 MFC drift (저측 offset)",
            "BCl3 supply line 부분 막힘 또는 line valve 부분 closed",
            "BCl3 cylinder 잔량 부족으로 outlet pressure 저하",
            "MFC seat 마모로 누설",
        ],
        "impact":   "BT 시간 ↑, native oxide / Al2O3 침투 불완전. ME endpoint detection 지연 가능.",
        "severity": "Medium",
        "response": {
            "immediate": [
                "현재 wafer SPC 결과 확인 → endpoint signal 지연/미검출 시 lot hold",
                "BCl3 cylinder 잔량(weight 또는 pressure) 즉시 확인",
                "BCl3 line manual valve open 상태 확인",
            ],
            "short_term": [
                "Cylinder 압력 정상이면 line filter / particle filter 막힘 점검",
                "MFC zero/span calibration 실행, 저측 offset 확인",
                "MFC seat 누설 의심 시 He leak check (line pressure decay test)",
            ],
            "verify": [
                "Calibration 후 BCl3 setpoint에서 30분 monitoring, Flow std nominal 수준 확인",
                "Monitor wafer 1매 → BT endpoint time이 historical 중심값 ±10% 내",
            ],
            "prevent": [
                "Cylinder swap 임계값을 보수적으로 조정 (잔량 X% 시점)",
                "Line filter PM 주기 단축 검토",
            ],
        },
    },

    # ═══════════════════════ Cl2 계열 ═══════════════════════
    "Cl2 +5": {
        "physical": "Cl2 main-etch gas setpoint이 nominal 대비 +5 sccm 초과.",
        "machine":  "Cl2 Flow가 nominal +~5 sccm.",
        "oes":      "Cl I(725.0 nm) ↑, Cl2 band ↑. AlCl reaction product 동반 ↑. Cl/Ar ratio ↑.",
        "rfm":      "전기음성 가스 증가로 electron density ↓ → impedance |Z| ↑. matching tune 이동.",
        "causes":   [
            "Cl2 MFC drift (고측)",
            "Cl2 MFC zero error",
            "MFC controller aging",
            "Cl2 supply pressure 상승",
        ],
        "impact":   "Main etch rate ↑, lateral etch 증가 → CD loss / undercut 위험. profile bowing 가능.",
        "severity": "Medium",
        "response": {
            "immediate": [
                "현재 lot CD inline 측정으로 spec out 여부 확인, out이면 hold",
                "Cl2 line GFM 데이터로 MFC reading vs 실제 flow 비교",
                "후속 wafer는 etch time 미세 단축 검토 (recipe feedback control, IBM Step 4)",
            ],
            "short_term": [
                "Cl2 MFC zero/span calibration (벤더 절차)",
                "Calibration으로 회복 안 되면 MFC 교체",
                "Cl2 supply regulator 출력압 확인",
            ],
            "verify": [
                "Calibration 후 zero check (commanded 0 sccm, reading <1% FS)",
                "Monitor wafer CD measurement → SPC 한계 내",
                "OES Cl I(725 nm) intensity가 nominal envelope 복귀",
            ],
            "prevent": [
                "Cl2 MFC calibration 주기 단축",
                "CD measurement 빈도 증가 (per-lot → per-batch)",
            ],
        },
    },

    "Cl2 -5": {
        "physical": "Cl2 setpoint이 nominal 대비 −5 sccm 부족.",
        "machine":  "Cl2 Flow가 nominal 대비 약 5 sccm 낮음.",
        "oes":      "Cl I(725.0 nm) ↓, Cl2 band ↓. AlCl product 감소.",
        "rfm":      "Electronegative species 감소로 electron density 일부 회복, impedance 변화 작음.",
        "causes":   [
            "Cl2 MFC drift (저측)",
            "Cl2 line 부분 막힘",
            "Cl2 cylinder outlet pressure 저하",
            "MFC seat 마모",
        ],
        "impact":   "Main ME etch rate ↓. CD on-target 유지 위해 etch time 연장 필요. throughput 영향.",
        "severity": "Medium",
        "response": {
            "immediate": [
                "ME endpoint signal 지연 모니터링, 일정 시간 초과 시 lot hold",
                "Cl2 cylinder 잔량 즉시 확인",
                "Cl2 line manual valve 상태 확인",
            ],
            "short_term": [
                "BCl3 -5와 동일: cylinder 정상 시 filter 막힘 → MFC zero/span calibration → MFC seat leak test 순",
                "Recovery 안 되면 MFC 교체",
            ],
            "verify": [
                "Cl2 setpoint 30분 안정성 monitoring",
                "Monitor wafer ME endpoint time SPC 내",
            ],
            "prevent": [
                "Cl2 cylinder swap 임계 보수화",
                "Endpoint detection 알고리즘 robustness 검토",
            ],
        },
    },

    "Cl2 -10": {
        "physical": "Cl2 setpoint이 nominal 대비 −10 sccm 부족 (Cl2 -5의 2배 결핍).",
        "machine":  "Cl2 Flow가 nominal 대비 약 10 sccm 낮음.",
        "oes":      "Cl I·Cl2 band 뚜렷이 ↓. AlCl product 약 절반 이하.",
        "rfm":      "Impedance가 nominal envelope을 명확히 벗어남.",
        "causes":   [
            "Cl2 MFC가 control loop 이탈",
            "Cl2 line 심한 막힘 또는 valve 부분 closed",
            "Cl2 cylinder 잔량 부족 또는 regulator 고장",
            "MFC sensor element 고장",
        ],
        "impact":   "ME etch rate 심하게 ↓. Etch stop 또는 endpoint mis-detection. wafer 폐기 가능.",
        "severity": "High",
        "response": {
            "immediate": [
                "즉시 lot hold (IBM Step 2: Halt). production 중단",
                "처리 중이던 wafer는 inline inspection 우선",
                "Cl2 cylinder 압력·잔량 즉시 확인, 부족 시 swap 우선",
            ],
            "short_term": [
                "Cylinder 정상이면 line 막힘 점검 (delta-P measurement at filter inlet/outlet)",
                "MFC bench test 또는 교체",
                "Cl2 line 전체 He purge → 청정도 확인",
            ],
            "verify": [
                "Dry-run 5회로 Cl2 Flow std 안정 확인",
                "Monitor wafer 3매 etch → CD, etch rate, endpoint 모두 SPC 내",
                "Chamber seasoning 2–4시간 (Cl2 chemistry 균형 회복)",
            ],
            "prevent": [
                "Cl2 line 정기 PM에 filter 압력 차이 모니터링 항목 추가",
                "MFC 사용 시간 기반 proactive 교체 정책 도입",
            ],
        },
    },

    # ═══════════════════════ He Chuck ═══════════════════════
    "He Chuck": {
        "physical": "Wafer backside He cooling pressure 이상. He chuck system 자체 결함 (setpoint deviation이 아닌 hardware 문제).",
        "machine":  "He Press 채널이 nominal envelope을 벗어나거나 std가 비정상적으로 큼. 다른 process gas 채널은 정상일 수 있음.",
        "oes":      "직접적 OES signature 약함. wafer 온도 변동 간접 영향으로 일부 product emission(AlCl 등) 변동 가능.",
        "rfm":      "Wafer/chuck capacitance 변화로 bias coupling 미세 변동. matching 일시적 흔들림 가능.",
        "causes":   [
            "Chuck O-ring/seal degradation으로 He leak (가장 흔함)",
            "Wafer mis-seating 또는 ESC clamp 이상",
            "He delivery line / regulator 이상",
            "He pressure transducer drift",
        ],
        "impact":   "Wafer 온도 불균일 → across-wafer CD uniformity 악화, selectivity drift. 심한 leak 시 chuck/chamber contamination.",
        "severity": "High (wafer-affecting + 챔버 contamination 가능성)",
        "response": {
            "immediate": [
                "즉시 lot hold + 처리 중 wafer 폐기 검토 (wafer thermal damage 가능성)",
                "Reverse-bias unchucking 시도, wafer 정상 release 여부 확인 (강제 lift-pin 금지)",
                "He flow rate 확인: ≥ 20 sccm이면 complete clamp failure로 판정 (CSMantech 기준)",
            ],
            "short_term": [
                "Chamber open, ESC 표면 시각 검사 + particle/contamination 확인",
                "ESC O-ring 교체 (Foamtec/MIT Labnetwork 권장 정기 교체 항목)",
                "He delivery line leak check (He leak detector 사용)",
                "ESC repair 또는 swap 결정 — ±5°C 이상 온도 deviation 시 repair 권장 (MDPI Electronics 11/6/880)",
            ],
            "verify": [
                "Clamp test (가압 5–10 kV, He back-pressure 2–20 Torr) 후 leak-by flow < 5 sccm 확인",
                "Dummy wafer 1매로 thermal uniformity 확인 (가능하면 OTMS 또는 fiber probe)",
                "Chamber seasoning 후 monitor wafer 1매로 CD uniformity SPC 내 확인",
            ],
            "prevent": [
                "ESC O-ring 교체 주기 단축 (사용 시간 기반)",
                "He leak rate trend chart SPC 도입 (climbing 시 proactive 대응)",
                "Wafer bow 사양 강화 (높은 bow는 leak-by 원인)",
            ],
        },
    },

    # ═══════════════════════ Pressure (Pr) 계열 ═══════════════════════
    "Pr +1": {
        "physical": "Chamber pressure setpoint이 nominal 대비 +1 mTorr 초과 (가벼운 excursion).",
        "machine":  "Pressure 채널이 nominal +~1 mTorr. throttle valve(Vat Valve) position이 보상을 위해 이동했을 가능성.",
        "oes":      "전반적 emission intensity 미세 변화. excited state 분포 미세 redistribute.",
        "rfm":      "Impedance 작은 시프트, harmonics 비율 미세 변화.",
        "causes":   [
            "Throttle valve calibration drift",
            "Pressure gauge (Baratron capacitance manometer) drift",
            "Pump capacity 미미한 저하",
            "Recipe step 직후 stabilization 시간 부족",
        ],
        "impact":   "Anisotropy 영향 미미. CD on-target 유지 가능. throughput 영향 거의 없음.",
        "severity": "Low",
        "response": {
            "immediate": [
                "현재 lot SPC 결과 확인, in-spec이면 계속 진행",
                "Pressure stabilization time이 recipe상 충분한지 확인 (typically ≥ 5초)",
            ],
            "short_term": [
                "Baratron zero check (chamber base pressure 상태에서 reading 확인)",
                "Throttle valve position이 nominal envelope 내인지 SPC chart로 확인",
            ],
            "verify": [
                "Monitor wafer 1매로 etch rate / CD 확인",
            ],
            "prevent": [
                "Baratron zero check 주기 정기화",
                "Throttle valve calibration PM 주기 검토",
            ],
        },
    },

    "Pr +2": {
        "physical": "Chamber pressure이 nominal 대비 +2 mTorr 초과.",
        "machine":  "Pressure ~+2 mTorr. Vat Valve가 더 닫힌 방향으로 이동했을 수 있음.",
        "oes":      "Collisional regime 강화 → dissociation 효율 감소, emission ratio 변동.",
        "rfm":      "Impedance 시프트가 Pr +1보다 분명.",
        "causes":   [
            "Throttle valve drift",
            "Pressure sensor calibration drift",
            "Pump aging / foreline conductance 변화",
            "Total gas flow 미세 증가의 압력 영향",
        ],
        "impact":   "Anisotropy 다소 저하, sidewall profile 약간 rounded. CD bias 변화 시작.",
        "severity": "Low–Medium",
        "response": {
            "immediate": [
                "Lot SPC 결과 확인. CD/profile 측정 결과 임계 근접 시 hold",
                "Vat Valve position log 확인",
            ],
            "short_term": [
                "Baratron 2-point calibration (zero + atm 또는 known reference)",
                "Throttle valve actuator response time 측정",
                "Pump exhaust pressure (foreline) check",
            ],
            "verify": [
                "Chamber base pressure가 spec 내 (전형적 < 10 μTorr)",
                "Monitor wafer 측정으로 CD bias 회귀 확인",
            ],
            "prevent": [
                "Pump PM 주기 검토 (oil 상태, bearing 음 등)",
                "Foreline cleaning 주기 점검",
            ],
        },
    },

    "Pr +3": {
        "physical": "Chamber pressure이 nominal 대비 +3 mTorr 초과 (Pr 양방향 중 가장 큰 excursion).",
        "machine":  "Pressure ~+3 mTorr. Vat Valve가 control range 상단 근처 가능.",
        "oes":      "Collisional regime 변화 뚜렷 → ion-to-radical ratio ↓. AlCl 등 product emission 감소.",
        "rfm":      "Impedance 명확히 시프트. matching tune이 envelope 경계 부근.",
        "causes":   [
            "Throttle valve actuator 이상",
            "Pump 성능 저하 / foreline restriction",
            "Pressure gauge drift",
            "Gas distribution baffle 막힘",
        ],
        "impact":   "Anisotropy 저하, sidewall bowing 가능, CD bias 명확. profile 품질 spec out 위험.",
        "severity": "Medium",
        "response": {
            "immediate": [
                "Lot hold",
                "Vat Valve가 full-open 근처면 pump 측 conductance 문제 시사 → pump 상태 우선 점검",
                "Vat Valve 중간 위치인데 pressure 못 잡으면 valve actuator 자체 문제 의심",
            ],
            "short_term": [
                "Throttle valve calibration (zero/full position) 재실행",
                "Pump performance test (base pressure 도달 시간 측정)",
                "Foreline / pump exhaust restriction 점검",
                "Baratron sensor 교체 검토 (drift 의심 시)",
            ],
            "verify": [
                "Base pressure 회복 확인",
                "Pressure step response 측정 (setpoint change → settle time)",
                "Monitor wafer 2–3매로 profile/CD 확인",
            ],
            "prevent": [
                "Pump rebuild 주기 모니터링",
                "Throttle valve actuator 정기 calibration",
            ],
        },
    },

    "Pr -2": {
        "physical": "Chamber pressure이 nominal 대비 −2 mTorr 부족.",
        "machine":  "Pressure ~−2 mTorr. Vat Valve가 더 열린 방향.",
        "oes":      "Lower pressure → mean free path ↑. emission intensity 다소 ↓이나 ion-rich 방향.",
        "rfm":      "Impedance 시프트(저측). plasma stability margin 감소 가능.",
        "causes":   [
            "Throttle valve drift (더 열림)",
            "Pressure gauge drift",
            "상류 결함으로 gas flow 부족",
            "Pump conductance 이례적 증가",
        ],
        "impact":   "Etch rate 다소 ↓, profile은 더 anisotropic해질 수 있으나 plasma instability risk.",
        "severity": "Low–Medium",
        "response": {
            "immediate": [
                "Gas total flow 정상 여부 확인 (MFC reading 합산)",
                "Lot SPC 결과로 spec out 여부 판단",
            ],
            "short_term": [
                "MFC 전체 zero/span check (특히 BCl3, Cl2)",
                "Throttle valve가 nominal보다 열려있는지 position log 확인",
                "Baratron zero/calibration check",
            ],
            "verify": [
                "Pressure setpoint stability test (±5% 내)",
                "Plasma ignition / sustain stability 확인 (RF reflected power 모니터링)",
            ],
            "prevent": [
                "Pressure SPC 한계 점검",
                "Plasma stability 지표(RF reflected power) trending",
            ],
        },
    },

    # ═══════════════════════ RF Bottom (Bias) 계열 ═══════════════════════
    # 출처: Coaxial Power FAQ, Advanced Energy 백서 — reflected power 발생 시
    # generator → cable → match → load 순으로 isolation.
    "RF +8": {
        "physical": "RF Bottom(bias) power가 nominal 대비 +8 W 초과.",
        "machine":  "RF Pwr / RF Btm Pwr가 nominal +~8 W. RF Tuner/Load 위치 미세 이동.",
        "oes":      "Higher bias → wafer 표면 ion bombardment ↑. sputtered product line(AlCl 등) 미세 ↑.",
        "rfm":      "RF Btm Pwr 변동과 동기. bias-side harmonics(S2, S3) 변화.",
        "causes":   [
            "RF bias generator drift",
            "Match network capacitor wear / drift",
            "Power forward/reflected sensor calibration error",
            "RF cable / connector degradation",
        ],
        "impact":   "Ion energy ↑ → vertical etch rate ↑, underlying layer selectivity ↓. photoresist erosion 미세 증가.",
        "severity": "Low–Medium",
        "response": {
            "immediate": [
                "Reflected power 채널 확인. forward의 5% 이하면 sensor drift, 5–20%면 match drift 의심",
                "Lot SPC 결과 확인 후 진행/hold 결정",
            ],
            "short_term": [
                "RF generator power calibration check (dummy load 50Ω 연결, 출력 측정)",
                "Match network capacitor position log 비교 (nominal envelope 내 여부)",
                "RF cable/connector torque check, 변색·산화 시각 점검",
            ],
            "verify": [
                "Calibration 후 forward/reflected power가 nominal envelope 복귀",
                "Monitor wafer 1매로 etch rate / selectivity 확인",
            ],
            "prevent": [
                "RF generator calibration 주기 확립",
                "Match network capacitor 사용 시간 트래킹",
            ],
        },
    },

    "RF +10": {
        "physical": "RF Bottom power가 nominal 대비 +10 W 초과 (+8보다 다소 심함).",
        "machine":  "RF Pwr가 nominal +~10 W. Tuner 이동 폭 더 큼.",
        "oes":      "Bias-driven product emission이 +8보다 뚜렷이 ↑.",
        "rfm":      "Harmonics 분포 변화 명확. matching이 envelope 경계 근처.",
        "causes":   [
            "RF generator output drift",
            "Match network 부품 aging",
            "Power sensor drift",
            "RF feed line impedance 변화",
        ],
        "impact":   "Vertical etch 가속, underlying stop layer 손상 risk. over-etch 위험.",
        "severity": "Medium",
        "response": {
            "immediate": [
                "Lot hold, CD/profile 확인",
                "Reflected power level 확인 (Advanced Energy 백서: >20% rated면 foldback)",
            ],
            "short_term": [
                "RF generator 50Ω dummy load 테스트로 generator 출력 자체 검증",
                "Generator 정상이면 match network 검사 — VVC(variable vacuum capacitor) 위치 nominal 대비 deviation 측정",
                "Match network 부품 교체 또는 generator 교체 결정",
            ],
            "verify": [
                "Calibration / 교체 후 plasma ignition 5회 연속 성공",
                "Forward power 안정성 ±2% 내, reflected < 5%",
                "Monitor wafer로 etch rate, selectivity 검증",
            ],
            "prevent": [
                "Match network VVC 사용 시간 한계 도입",
                "RF reflected power SPC chart 도입",
            ],
        },
    },

    "RF -12": {
        "physical": "RF Bottom power가 nominal 대비 −12 W 부족 (RF 계열 중 가장 큰 결핍).",
        "machine":  "RF Pwr가 nominal 대비 약 12 W 낮음. RF Btm Rfl Pwr(reflected)이 평소보다 클 가능성.",
        "oes":      "Ion bombardment 부족 → sputtered product emission ↓. polymer accumulation signature(C 기반) 가능.",
        "rfm":      "Bias-side harmonics 약화. matching 위치 이상. Reflected power가 elevated이면 |Z| mismatch 강력 의심.",
        "causes":   [
            "RF generator output 저하",
            "Match network mismatch / 부품 고장 (Coaxial Power FAQ: forward/reflected 이상의 대부분이 match 또는 cable에서 발생)",
            "RF cable 손상 또는 connector 접촉 불량",
            "Power sensor drift",
        ],
        "impact":   "Vertical etch 부족 → 잔류물 risk, profile 불완전. ME endpoint 지연/미검출. polymer-rich 환경.",
        "severity": "High",
        "response": {
            "immediate": [
                "Lot hold (Halt)",
                "RF Reflected Power 즉시 확인 — Advanced Energy/Coaxial 표준에 따라 >20% rated이면 자동 foldback 가능성",
                "Wafer는 잔류물 inspection 후 폐기/재가공 결정",
            ],
            "short_term": [
                "Isolation 절차 (Coaxial Power FAQ): generator → coax/cable → match network → chamber load 순",
                "1) Generator: 50Ω dummy load 연결 → 출력 측정 (정상이면 generator OK)",
                "2) Cable: 시각 검사, connector torque 재체결, 가능하면 swap",
                "3) Match: capacitor 위치 manual 조정으로 minimum reflected 확인",
                "4) Load: chamber 상태 — wall deposit 누적 시 wet clean 검토",
            ],
            "verify": [
                "수리 후 plasma ignition 10회 연속 성공",
                "Forward power 안정성, reflected < 5%",
                "Chamber seasoning 후 monitor wafer 3매로 etch rate, profile, residue inspection",
            ],
            "prevent": [
                "RF Reflected Power SPC 도입, 한계 초과 시 즉시 alarm",
                "RF cable 정기 시각 점검 PM 항목 추가",
            ],
        },
    },

    # ═══════════════════════ TCP (Source) 계열 ═══════════════════════
    # TCP는 source power로 plasma density 결정. 대응 절차는 RF 계열과 유사하나
    # match network가 TCP 전용 회로이며 dual-frequency 시스템(MDPI 2021)에서는
    # 독립 매칭 필요.
    "TCP +10": {
        "physical": "TCP Top(source) power가 nominal 대비 +10 W 초과 (가벼운 excursion).",
        "machine":  "TCP Top Pwr가 nominal +~10 W. TCP Tuner/Load 미세 이동.",
        "oes":      "Plasma density 미세 ↑ → 전반적 emission intensity ↑.",
        "rfm":      "TCP 회로 측 impedance 미세 시프트.",
        "causes":   [
            "TCP RF generator drift",
            "TCP match network drift",
            "Power sensor drift",
            "Coil aging",
        ],
        "impact":   "Reactive species 미세 ↑, etch rate 약간 ↑. CD/profile 영향 작음.",
        "severity": "Low",
        "response": {
            "immediate": [
                "Lot SPC 결과 확인, in-spec이면 계속 진행",
                "TCP Rfl Pwr 채널 확인 (정상 < 5% forward)",
            ],
            "short_term": [
                "RF +8과 동일한 isolation 절차 적용 (generator → match → load)",
                "TCP generator dummy load test",
            ],
            "verify": [
                "Monitor wafer로 etch rate 확인",
            ],
            "prevent": [
                "TCP generator calibration 주기 확립",
            ],
        },
    },

    "TCP +20": {
        "physical": "TCP power가 nominal 대비 +20 W 초과.",
        "machine":  "TCP Top Pwr 약 +20 W.",
        "oes":      "Overall emission intensity ↑ 명확. 모든 species line 동반 ↑.",
        "rfm":      "TCP-side impedance / load shift 명확.",
        "causes":   [
            "TCP RF generator drift",
            "Match network capacitor wear",
            "Coil의 dielectric / material change",
            "Power calibration error",
        ],
        "impact":   "Etch rate ↑, chemical etch component 증가. lateral etch / CD loss 시작.",
        "severity": "Medium",
        "response": {
            "immediate": [
                "Lot CD inline 측정, spec out 시 hold",
                "TCP Rfl Pwr 확인",
            ],
            "short_term": [
                "RF +10과 동일 절차: generator dummy load test → match network 검사 → coil 상태 점검",
                "Coil(antenna) 시각 점검 — 변색, 균열, dielectric 손상 확인",
            ],
            "verify": [
                "Calibration 후 plasma ignition 안정성",
                "Monitor wafer로 CD, etch rate 검증",
            ],
            "prevent": [
                "Coil PM 주기 검토 (사용 시간 또는 cycle count 기반)",
            ],
        },
    },

    "TCP +30": {
        "physical": "TCP power가 nominal 대비 +30 W 초과.",
        "machine":  "TCP Top Pwr 약 +30 W.",
        "oes":      "Emission intensity 큰 폭 ↑. plasma over-dense 시그니처(metastable line 비선형 증가).",
        "rfm":      "Harmonics 분포 변화 명확. matching이 envelope 경계 또는 벗어남.",
        "causes":   [
            "TCP RF generator significant drift",
            "Match controller 이상으로 over-coupling",
            "Coil short 또는 partial dielectric failure",
            "Power sensor 큰 calibration error",
        ],
        "impact":   "Etch rate 큰 폭 ↑, profile widening, selectivity 저하. wafer-level CD spec out 가능.",
        "severity": "Medium–High",
        "response": {
            "immediate": [
                "즉시 lot hold",
                "TCP RF off 검토 (안전 절차상 over-coupling 의심 시)",
                "처리 중 wafer는 inline CD/profile 검사",
            ],
            "short_term": [
                "TCP generator output 정밀 측정",
                "Match network 부품 (특히 VVC) 위치/상태 정밀 점검",
                "Coil 분해 점검 — dielectric layer 손상 확인",
                "Generator 또는 match unit swap 결정",
            ],
            "verify": [
                "Swap 후 plasma 안정성 — ignition 10회 + 30분 연속 안정",
                "TCP impedance(RFM) nominal envelope 복귀",
                "Monitor wafer 3매로 etch rate, CD, profile 모두 SPC 내",
            ],
            "prevent": [
                "Coil 사용 시간 한계 도입 (proactive 교체)",
                "TCP Rfl Pwr SPC 도입",
            ],
        },
    },

    "TCP +50": {
        "physical": "TCP power가 nominal 대비 +50 W 초과 (TCP 양방향 중 가장 큰 excursion).",
        "machine":  "TCP Top Pwr 약 +50 W. TCP Rfl Pwr 변화 가능.",
        "oes":      "Emission intensity 전반적 큰 폭 상승. plasma density 포화 영역 근처 비선형 응답.",
        "rfm":      "Impedance가 nominal range 밖. Match tune이 한계 부근 또는 fault.",
        "causes":   [
            "TCP RF generator stuck-high / control loop 실패",
            "Match network 심각한 부품 고장",
            "Coil의 ground path 이상",
            "Power monitoring 회로 fault",
        ],
        "impact":   "심한 over-etch, underlying layer 손상, profile 완전 변형. wafer 폐기 가능성 높음. equipment safety check 필요.",
        "severity": "Critical",
        "response": {
            "immediate": [
                "EMERGENCY: TCP RF immediate shutdown 검토",
                "Lot hold + 처리 중 wafer 폐기 가정",
                "EQ engineer 즉시 호출, safety 절차 우선 (Power monitoring fault면 실제 power가 더 클 수 있음)",
                "Chamber over-temp / RF arcing 흔적 점검 (chamber open 전 시각 점검)",
            ],
            "short_term": [
                "TCP generator 출력 자체 calibration (dummy load 50Ω)",
                "Match network 전체 점검 또는 unit swap",
                "Coil ground path 검사 (DC resistance 측정)",
                "Power monitoring 회로 verification (independent meter로 비교)",
                "필요 시 chamber wet clean (arcing이 chamber wall에 손상 줬을 가능성)",
            ],
            "verify": [
                "수리 후 dummy wafer로 plasma ignition 20회 + 1시간 연속 안정 확인",
                "TCP Rfl Pwr < 3% forward",
                "Monitor wafer 5매로 etch rate, CD, profile, defect density 모두 SPC 내",
                "Chamber seasoning 4시간 이상 권장",
            ],
            "prevent": [
                "TCP generator 사용 시간 기반 proactive 교체 정책",
                "TCP Power upper SPC 한계 강화 (hard interlock)",
                "RF system 정기 health check PM 추가",
            ],
        },
    },

    "TCP -15": {
        "physical": "TCP power가 nominal 대비 −15 W 부족.",
        "machine":  "TCP Top Pwr가 nominal 대비 약 15 W 낮음.",
        "oes":      "Overall emission intensity ↓ 명확. 모든 species line 동반 ↓.",
        "rfm":      "TCP-side impedance 시프트. Reflected power가 elevated일 가능성.",
        "causes":   [
            "TCP RF generator drift (저측)",
            "Match network drift",
            "Coil / feed-thru 손상으로 power loss",
            "Power calibration error",
        ],
        "impact":   "Plasma density ↓, etch rate ↓. ME 시간 ↑ 필요. plasma stability 저하 가능.",
        "severity": "Medium",
        "response": {
            "immediate": [
                "Lot SPC 결과로 hold 여부 판단",
                "TCP Rfl Pwr 확인 (elevated면 match/coil 문제)",
            ],
            "short_term": [
                "RF -12와 동일 isolation 절차: generator → cable → match → load",
                "Coil feed-thru 점검 (ceramic 손상, brazing crack 등)",
                "Power sensor calibration",
            ],
            "verify": [
                "Forward power 안정 ±2%, reflected < 5%",
                "Monitor wafer로 etch rate 확인",
            ],
            "prevent": [
                "Coil feed-thru PM 점검 항목 추가",
                "TCP Power SPC 한계 검토",
            ],
        },
    },

    "TCP -20": {
        "physical": "TCP power가 nominal 대비 −20 W 부족 (TCP 음방향 가장 큰 결핍).",
        "machine":  "TCP Top Pwr가 nominal 대비 약 20 W 낮음.",
        "oes":      "Emission intensity 큰 폭 ↓. plasma extinction 근접 condition 가능.",
        "rfm":      "Impedance ↑, matching 한계 부근. Reflected power elevated 가능.",
        "causes":   [
            "TCP RF generator output 큰 폭 저하",
            "Match network 심각한 mismatch",
            "Coil / feed-thru 손상",
            "Power monitoring fault로 실제 power 미달",
        ],
        "impact":   "Etch rate 큰 폭 ↓, possible etch stop 또는 plasma instability. wafer 폐기 가능.",
        "severity": "High",
        "response": {
            "immediate": [
                "Lot hold",
                "Plasma ignition 안정성 점검 — 불안정 시 즉시 RF off",
                "Wafer는 endpoint 도달 여부 따라 폐기/재가공 분류",
            ],
            "short_term": [
                "RF -12와 동일 isolation 절차",
                "Coil 시각 분해 점검 (brazing, dielectric, deposit 확인)",
                "Generator output 정밀 측정 후 교체 결정",
                "필요 시 match unit swap",
            ],
            "verify": [
                "Plasma ignition 10회 + 30분 안정",
                "TCP Rfl Pwr < 5%",
                "Monitor wafer 3매로 검증",
            ],
            "prevent": [
                "Coil & generator 사용 시간 트래킹 강화",
                "TCP Power lower SPC 한계 강화",
            ],
        },
    },
}

assert len(FAULT_KNOWLEDGE) == 20, "결함 개수 점검: 20개여야 함"


# ---------------------------------------------------------------------------
# 4) 프롬프트 빌더
# ---------------------------------------------------------------------------
def _format_stats(stats: dict) -> str:
    if not stats:
        return "(센서 통계가 제공되지 않음)"
    if not any(k in stats for k in ("MACHINE", "OES", "RFM")):
        return "\n".join(f"  {k}: {m:.4g} ± {s:.4g}" for k, (m, s) in stats.items())
    lines = []
    for group in ("MACHINE", "OES", "RFM"):
        if group in stats and stats[group]:
            lines.append(f"[{group}]")
            for k, (m, s) in stats[group].items():
                lines.append(f"  {k}: {m:.4g} ± {s:.4g}")
    return "\n".join(lines) if lines else "(센서 통계가 제공되지 않음)"


def _format_response_list(items: list[str], indent: str = "    ") -> str:
    return "\n".join(f"{indent}{i+1}. {item}" for i, item in enumerate(items))


def build_prompt(fault_label: str, stats: dict | None = None) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt).
    """
    if fault_label not in FAULT_KNOWLEDGE:
        raise KeyError(
            f"Unknown fault '{fault_label}'. Available: {sorted(FAULT_KNOWLEDGE)}"
        )

    card = FAULT_KNOWLEDGE[fault_label]
    resp = card["response"]
    causes_block = "\n".join(f"    {i+1}. {c}" for i, c in enumerate(card["causes"]))
    stats_block = _format_stats(stats or {})

    user = USER_TEMPLATE.format(
        fault_label=fault_label,
        stats_block=stats_block,
        physical=card["physical"],
        machine=card["machine"],
        oes=card["oes"],
        rfm=card["rfm"],
        causes_block=causes_block,
        impact=card["impact"],
        severity=card["severity"],
        resp_immediate=_format_response_list(resp["immediate"]),
        resp_short=_format_response_list(resp["short_term"]),
        resp_verify=_format_response_list(resp["verify"]),
        resp_prevent=_format_response_list(resp["prevent"]),
    )
    return SYSTEM_PROMPT, user


# ---------------------------------------------------------------------------
# 5) 사용 예시
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    example_stats = {
        "MACHINE": {
            "BCl3 Flow":     (751.1, 0.32),
            "Cl2 Flow":      (124.5, 0.18),
            "Pressure":      ( 10.02, 0.05),
            "He Press":      (  8.01, 0.02),
            "RF Btm Pwr":    (250.3, 1.10),
            "RF Btm Rfl Pwr":( 35.2, 4.10),                 # 비정상적으로 큼
            "TCP Top Pwr":   (450.2, 2.20),
            "TCP Rfl Pwr":   ( 12.5, 1.80),
        },
        "OES": {
            "272.2": (415.3, 12.4),
            "284.6": (612.8, 18.9),
            "725.0": (980.6, 25.7),
        },
        "RFM": {
            "S1V1": (146.2, 0.85),
            "S1I1": ( 12.3, 0.07),
            "S2V1": ( 41.7, 0.55),
        },
    }
    sys_p, user_p = build_prompt("RF -12", example_stats)
    print("===== SYSTEM PROMPT =====\n")
    print(sys_p)
    print("\n\n===== USER PROMPT =====\n")
    print(user_p)
