from fastapi import Header, HTTPException, status

from .config import get_settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    """API-key auth for the pilot.

    Production note: replace with Entra ID bearer-token validation. Keeping the
    check in one dependency means that swap touches this file only.
    """
    settings = get_settings()
    if not x_api_key or x_api_key not in settings.api_key_set:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Provide a valid X-API-Key header.",
        )
    return x_api_key
