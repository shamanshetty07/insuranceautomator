"""Mapping between ExtractedClaim (the contract) and report_claims rows (storage). Pure."""

import uuid
from collections.abc import Sequence
from typing import Any, Protocol

from app.schemas.claims import ClaimExtractionResult, ClaimType, ExtractedClaim


class _ClaimRow(Protocol):
    claim_id: str
    text: str
    claim_type: ClaimType | str
    damage_type: str | None
    image_ids: Sequence[str]
    prediction_ids: Sequence[str]
    evidence: Sequence[dict[str, Any]]


def claim_row_values(claim: ExtractedClaim, position: int) -> dict[str, Any]:
    """Column values for one report_claims row (without ids/foreign keys)."""
    dumped = claim.model_dump(mode="json")
    return {
        "position": position,
        "claim_id": claim.claim_id,
        "text": claim.text,
        "claim_type": claim.claim_type,
        "damage_type": claim.damage_type,
        "image_ids": dumped["image_ids"],
        "prediction_ids": dumped["prediction_ids"],
        "evidence": dumped["evidence"],
    }


def claim_from_row(row: _ClaimRow) -> ExtractedClaim:
    claim_type = row.claim_type.value if isinstance(row.claim_type, ClaimType) else row.claim_type
    return ExtractedClaim.model_validate(
        {
            "claim_id": row.claim_id,
            "text": row.text,
            "claim_type": claim_type,
            "damage_type": row.damage_type,
            "image_ids": list(row.image_ids),
            "prediction_ids": list(row.prediction_ids),
            "evidence": list(row.evidence),
        }
    )


def claims_from_rows(report_id: uuid.UUID, rows: Sequence[_ClaimRow]) -> ClaimExtractionResult:
    """Rows must already be in position order (GeneratedReport.claims is)."""
    return ClaimExtractionResult(report_id=report_id, claims=[claim_from_row(r) for r in rows])
