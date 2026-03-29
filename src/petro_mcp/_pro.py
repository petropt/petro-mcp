import os

def is_pro() -> bool:
    """Check if Petropt Pro API key is set."""
    key = os.environ.get("PETROSUITE_API_KEY", "")
    return bool(key and key.startswith("psp_") and len(key) >= 32)
