"""Utility helpers for Crucible challenge interaction."""

import base64
import io

import requests
from PIL import Image


def submit_flag(
    adversarial_image: Image.Image,
    challenge_url: str,
    crucible_url: str,
    challenge: str,
    api_key: str,
) -> str | None:
    """Submit an adversarial image to the Crucible challenge and verify the flag.

    Args:
        adversarial_image: PIL image to submit.
        challenge_url:     Base URL of the challenge endpoint.
        crucible_url:      Base URL of the Crucible platform.
        challenge:         Challenge slug (e.g. "granny").
        api_key:           Dreadnode API key.

    Returns:
        The flag string if successful, None otherwise.
    """
    # Encode image as base64 PNG
    buf = io.BytesIO()
    adversarial_image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    # Submit to challenge endpoint
    result = requests.post(
        f"{challenge_url}/submit",
        headers={"X-API-Key": api_key},
        json={"data": b64},
    ).json()

    if "flag" not in result:
        output = result.get("output", [])
        sorted_output = sorted(output, key=lambda x: x[0], reverse=True)[:5]
        print("No flag returned. Server predictions:")
        for prob, label in sorted_output:
            print(f"  {label:30s} {prob:.6f}")
        print("\nThe image may not meet the L2 distance constraint.")
        return None

    flag = result["flag"]

    # Verify with Crucible platform
    submit_url = f"{crucible_url}/api/challenges/{challenge}/submit-flag"
    payload = {"challenge": challenge, "flag": flag}
    resp = requests.post(submit_url, headers={"X-API-Key": api_key}, json=payload)

    if resp.status_code == 200 and resp.json().get("correct") is True:
        print("Flag verified correct!")
    elif resp.status_code == 200:
        print("Flag was incorrect. Keep trying!")
    else:
        print(f"Error submitting flag: {resp.text}")

    return flag
