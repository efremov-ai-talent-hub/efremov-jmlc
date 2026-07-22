"""Seller identity used by prompts.

The analysers need *some* token for the selling side: ASR takes it as a
vocabulary hint, and the speaker-role detector keys off corporate-greeting
phrases built from it. Hardcoding a real company into prompt literals would
bake a specific client into this repository, so the default here is generic —
this stand ships with no real client.

SELLER_BRAND does NOT reach every prompt. It drives the ASR hint
(ai/transcription/transcriber.py) and the role detector
(ai/reports/call_v3/manager.py); the analyser prompts under ai/reports/ carry
the name inline because Russian declines it (Девелопера, Девелоперу) and naive
interpolation would produce broken grammar in instructions the model follows
closely. Setting SELLER_BRAND alone therefore yields a split brain: ASR and the
detector say one name while the graders still say another. Renaming the seller
means editing those prompts too — grep for the default value.
"""

from __future__ import annotations

import os

SELLER_BRAND: str = os.getenv("SELLER_BRAND", "Девелопер")


def seller_greeting_markers() -> tuple[str, ...]:
    """Corporate-greeting phrases that only the selling side says.

    Derived from the brand rather than listed literally so that overriding
    SELLER_BRAND keeps the speaker-role detector working.
    """
    brand = SELLER_BRAND.lower()
    return (
        f"компания {brand}",
        f"{brand} приветствует",
        f"приветствую вас от команды {brand}",
    )
