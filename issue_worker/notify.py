"""

Design:
- NotificationChannel: abstract interface, one method (send).
- SlackNotifier: real implementation, posts to a Slack incoming webhook.
- NullNotifier: no-op, used automatically when SLACK_WEBHOOK_URL isn't set..
- get_notifier(): factory — picks Slack or Null based on env config..
- notify_escalation(record): the actual entry point, called from escalate.py.

A Slack outage or missing webhook must never crash the pipeline — send()
always returns bool and swallows its own request errors.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


logger = logging.getLogger(__name__)


class NotificationChannel(ABC):
    @abstractmethod
    def send(self, event: dict[str, Any]) -> bool:
        """Send a notification event. Returns True on success, False on
        failure. Must never raise — callers treat this as fire-and-forget."""
        raise NotImplementedError


class SlackNotifier(NotificationChannel):
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, event: dict[str, Any]) -> bool:
        text = self._format(event)
        try:
            resp = requests.post(
                self.webhook_url,
                json={"text": text},
                timeout=5,
            )
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.warning("SlackNotifier: send failed (%s), continuing", e)
            return False

    _MAX_DETAIL_CHARS = 500

    @classmethod
    def _format(cls, event: dict[str, Any]) -> str:
        detail = str(event.get("detail", "(none)"))
        if len(detail) > cls._MAX_DETAIL_CHARS:
            detail = detail[: cls._MAX_DETAIL_CHARS] + "... (truncated)"
        return (
            f":rotating_light: *Issue Escalated*\n"
            f"*source_id:* `{event.get('source_id')}`\n"
            f"*category:* `{event.get('category')}`\n"
            f"*failure_reason:* `{event.get('failure_reason')}`\n"
            f"*detail:* {detail}"
        )


class NullNotifier(NotificationChannel):
    def send(self, event: dict[str, Any]) -> bool:
        logger.info("NullNotifier: SLACK_WEBHOOK_URL not set, skipping send. event=%s", event)
        return True


def get_notifier() -> NotificationChannel:
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if webhook_url:
        return SlackNotifier(webhook_url)
    return NullNotifier()


def notify_escalation(record: dict[str, Any]) -> bool:
    """Entry point called from escalate.py after an escalation record
    is built. record is expected to contain at least: source_id,
    category, failure_reason, detail."""
    notifier = get_notifier()
    return notifier.send(record)
