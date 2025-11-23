"""Privacy & ethics guardrails.

This module enforces high-level ethics constraints so that downstream agents avoid
using sensitive personal attributes without explicit consent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ConsentRecord:
    user_id: str
    consented_attributes: List[str]


class PrivacyGuard:
    def __init__(self, allowed_sensitive_attributes: List[str]) -> None:
        self.allowed = set(allowed_sensitive_attributes)
        self.explicit_consent: Dict[str, ConsentRecord] = {}

    def register_consent(self, user_id: str, attributes: List[str]) -> None:
        filtered = [attr for attr in attributes if attr in self.allowed]
        self.explicit_consent[user_id] = ConsentRecord(user_id=user_id, consented_attributes=filtered)

    def assert_safe(self, user_id: str, requested_attributes: List[str]) -> None:
        record = self.explicit_consent.get(user_id)
        if not record:
            raise PermissionError(
                "User consent is required before accessing sensitive attributes."
            )
        disallowed = [attr for attr in requested_attributes if attr not in record.consented_attributes]
        if disallowed:
            raise PermissionError(
                f"Requested attributes {disallowed} exceed granted consent for user={user_id}."
            )


# Example usage (wired by coordinator)
privacy_guard = PrivacyGuard(allowed_sensitive_attributes=["socio_economic_bracket", "region"])
