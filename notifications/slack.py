import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()


class SlackNotifier:
    def __init__(self):
        self.webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        self.bot_token = os.getenv("SLACK_BOT_TOKEN")
        self.channel_id = os.getenv("SLACK_CHANNEL_ID")

    def is_configured(self) -> bool:
        return bool(self.webhook_url or (self.bot_token and self.channel_id))

    def _effective_webhook(self, override: str | None) -> str | None:
        if override:
            return override
        return self.webhook_url

    def send_alert(
        self,
        fault_name,
        run_name,
        mse,
        confidence,
        explanation=None,
        recommendation=None,
        root_cause_sensors=None,
        equipment_id=None,
        event_time=None,
        alert_stage="immediate",
        webhook_url: str | None = None,
        username: str | None = None,
    ) -> bool:
        severity_label, severity_detail = self._severity_label_and_detail(confidence, mse)
        primary_sensor = self._primary_root_cause_sensor(root_cause_sensors)
        recommendation = self._clean_recommendation(recommendation)
        primary_cause = self._extract_primary_cause(recommendation, explanation, fault_name, primary_sensor)
        action_items = self._format_action_items(recommendation, primary_sensor, fault_name)

        payload = {
                    "username": username or "Semiconductor Guardian",
                    "icon_emoji": ":warning:",
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f":rotating_light: *[설비 이상]* {equipment_id or 'Unknown'} (Run: {run_name})",
                            },
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": (
                                    f"*발생 시간:* {event_time or 'Now'}\n"
                                    f"*이상 내용:* {fault_name}\n"
                                    f"*심각도:* {severity_label} ({severity_detail})\n"
                                    f"*원인 센서:* `{primary_sensor or 'Unknown'}`\n"
                                    f"*최우선 추정 원인:* {primary_cause}"
                                ),
                            },
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": action_items,
                            },
                        },
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": "(단기/장기 조치 사항은 사내 SOP를 참고해 주세요)",
                                }
                            ],
                        },
                    ],
                }

        webhook = self._effective_webhook(webhook_url)
        try:
            if webhook:
                encoded_data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
                response = requests.post(
                    webhook,
                    data=encoded_data,
                    headers={"Content-Type": "application/json; charset=utf-8"},
                    timeout=10,
                )
                response.raise_for_status()
                print(f"[Slack] Alert sent by webhook for {run_name}")
                return True

            if self.bot_token and self.channel_id:
                combined_payload = {"channel": self.channel_id, **payload}
                encoded_data = json.dumps(combined_payload, ensure_ascii=False).encode('utf-8')
                response = requests.post(
                    "https://slack.com/api/chat.postMessage",
                    data=encoded_data,
                    headers={
                        "Authorization": f"Bearer {self.bot_token}",
                        "Content-Type": "application/json; charset=utf-8"
                    },
                    timeout=10,
                )
                response.raise_for_status()
                body = response.json()
                if not body.get("ok"):
                    print(f"[Slack] Failed to send alert: {body.get('error', 'unknown_error')}")
                    return False
                print(f"[Slack] Alert sent by bot token for {run_name}")
                return True

            print("[Slack] No webhook URL or bot token/channel found. Skipping alert.")
            return False
        except Exception as exc:
            print(f"[Slack] Error sending alert: {exc}")
            return False

    @staticmethod
    def _trim_text(value, limit: int) -> str:
        text = str(value).strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    @staticmethod
    def _primary_root_cause_sensor(root_cause_sensors) -> str | None:
        if not root_cause_sensors:
            return None

        if isinstance(root_cause_sensors, (list, tuple)):
            for item in root_cause_sensors:
                if isinstance(item, dict):
                    sensor = item.get("sensor") or item.get("name") or item.get("feature")
                    if sensor:
                        return str(sensor)
                elif item:
                    return str(item)
            return None

        return str(root_cause_sensors)

    @staticmethod
    def _clean_recommendation(recommendation):
        if not recommendation:
            return None

        text = str(recommendation).strip()
        unavailable_markers = (
            "Neo4j/RAG recommendation unavailable",
            "Failed to DNS resolve",
            "getaddrinfo failed",
            "ServiceUnavailable",
            "Failed to establish a new connection",
        )
        if any(marker in text for marker in unavailable_markers):
            return None
        return text

    @staticmethod
    def _extract_primary_cause(recommendation, explanation, fault_name, primary_sensor=None) -> str:
        if recommendation:
            first_line = str(recommendation).strip().splitlines()[0].strip()
            if first_line:
                return first_line
        if primary_sensor:
            return f"AI 진단 결과, {primary_sensor} 센서의 변동이 주요 결함 원인으로 분석되었습니다."
        if explanation:
            first_line = str(explanation).strip().splitlines()[0].strip()
            if first_line and SlackNotifier._clean_recommendation(first_line):
                return first_line
        return fault_name or "원인 미확정"

    @staticmethod
    def _format_action_items(recommendation, primary_sensor=None, fault_name=None) -> str:
        if not recommendation:
            sensor = primary_sensor or "대표 원인 센서"
            return (
                "*현장 즉시 조치 사항*\n"
                f"1. `{sensor}` 센서의 실시간 값과 최근 추세를 정상 기준과 비교하세요.\n"
                "2. 해당 센서의 케이블, 통신, 캘리브레이션 상태와 레시피 세트포인트 변경 이력을 확인하세요.\n"
                "3. 동일 이상이 반복되면 설비를 Hold 상태로 유지하고 담당 엔지니어에게 계측 계통 점검을 요청하세요."
            )

        lines = [line.strip() for line in str(recommendation).splitlines() if line.strip()]
        if not lines:
            return SlackNotifier._format_action_items(None, primary_sensor, fault_name)

        formatted = "*현장 즉시 조치 사항*\n"
        for idx, line in enumerate(lines[:4], start=1):
            if line.startswith(('-', '*')):
                line = line[1:].strip()
            formatted += f"{idx}. {line}\n"
        return formatted

    @staticmethod
    def _severity_label_and_detail(confidence, mse) -> tuple[str, str]:
        if confidence is None:
            return "Unknown", "위험도 판단 불가"

        try:
            score = float(confidence)
        except (TypeError, ValueError):
            return "Unknown", "위험도 판단 불가"

        if score >= 0.90:
            return "High", "Wafer 폐기 위험: 높음"
        if score >= 0.70:
            return "Medium-High", "Wafer 폐기 위험: 중간 이상"
        if score >= 0.40:
            return "Low-Medium", "Wafer 폐기 위험: 낮음"
        return "Low", "Wafer 폐기 위험: 매우 낮음"


if __name__ == "__main__":
    notifier = SlackNotifier()
    notifier.send_alert("MANUAL TEST", "RUN_TEST_001", 0.85, 0.45, "This is a test explanation.")
