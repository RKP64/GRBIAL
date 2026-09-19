"""Sign-in, user records and an audit trail.

Runs alongside the existing API-key auth rather than replacing it. A caller may
present either an `Authorization: Bearer <jwt>` header or an `X-API-Key` header,
so machine-to-machine integrations built against the keys keep working while
people sign in.

Users and audit events go to MongoDB when a connection string is configured, and
to files under the data directory when it is not. The file path exists so a
developer can run the platform with nothing else installed; it is not suitable
for more than one instance, because two containers would each keep their own
copy.

The JWT and password hashing are implemented on the standard library rather than
pulling in pyjwt and passlib. Both are small, well-specified pieces, and the
alternative is two more dependencies to keep patched in an air-gapped
deployment.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import Header, HTTPException, Request

from .access import ALL_DOMAINS, Principal, Role
from .config import get_settings

log = logging.getLogger(__name__)

ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 480  # a working day; long enough not to interrupt


# ------------------------------------------------------------------ secret

def _secret() -> str:
    """Signing key for tokens.

    Falls back to the first configured API key so a development instance works
    without extra setup. In any deployment where sessions should survive a
    restart or span replicas, set JWT_SECRET — otherwise the fallback changes
    with the key list and every session is invalidated.
    """
    s = get_settings()
    configured = getattr(s, "jwt_secret", "") or ""
    if configured:
        return configured
    keys = sorted(s.api_key_set) if s.api_key_set else []
    if keys:
        return keys[0]
    log.warning("No JWT_SECRET set; using a development default.")
    return "kg-platform-development-secret"


# ------------------------------------------------------------------ passwords

PBKDF2_ITERATIONS = 200_000
# Records written before the hash carried its parameters used this count. They
# are still valid; they simply predate the format.
LEGACY_ITERATIONS = 100_000


def hash_password(password: str) -> str:
    """Store the parameters alongside the digest.

    Written as `pbkdf2$<iterations>$<salt>$<digest>` so the cost can be raised
    later without invalidating every existing password. The earlier two-part
    form is still read, which is what stops a stored account breaking when this
    changes.
    """
    salt = uuid.uuid4().hex
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(),
                                 PBKDF2_ITERATIONS)
    return f"pbkdf2${PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    if not stored:
        return False
    parts = stored.split("$")
    if len(parts) == 4 and parts[0] == "pbkdf2":
        try:
            iterations = int(parts[1])
        except ValueError:
            return False
        salt, digest = parts[2], parts[3]
    elif len(parts) == 2:
        # salt$digest — the original form, fixed iteration count.
        iterations, salt, digest = LEGACY_ITERATIONS, parts[0], parts[1]
    else:
        return False
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(),
                                iterations)
    return hmac.compare_digest(check.hex(), digest)


# ------------------------------------------------------------------ tokens

def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def create_token(payload: dict[str, Any]) -> str:
    secret = _secret()
    header = _b64e(json.dumps({"alg": ALGORITHM, "typ": "JWT"}).encode())
    now = int(time.time())
    body = _b64e(json.dumps({**payload, "iat": now,
                             "exp": now + TOKEN_EXPIRE_MINUTES * 60,
                             "jti": uuid.uuid4().hex}).encode())
    signature = hmac.new(secret.encode(), f"{header}.{body}".encode(),
                         hashlib.sha256).digest()
    return f"{header}.{body}.{_b64e(signature)}"


def decode_token(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Malformed token.")
    header, body, signature = parts
    expected = hmac.new(_secret().encode(), f"{header}.{body}".encode(),
                        hashlib.sha256).digest()
    # Constant-time compare: a short-circuiting comparison here leaks the
    # signature a byte at a time to anyone willing to measure.
    if not hmac.compare_digest(expected, _b64d(signature)):
        raise ValueError("Token signature does not match.")
    payload = json.loads(_b64d(body))
    if payload.get("exp", 0) < int(time.time()):
        raise ValueError("Session has expired. Sign in again.")
    return payload


# ------------------------------------------------------------------ users

@dataclass
class User:
    username: str
    email: str = ""
    password_hash: str = ""
    display_name: str = ""
    role: str = "editor"
    domains: list[str] = field(default_factory=lambda: [ALL_DOMAINS])
    is_active: bool = True
    created_at: str = ""
    last_login: str = ""

    def safe_dict(self) -> dict[str, Any]:
        """Everything except the hash. Used for anything leaving the server."""
        return {k: v for k, v in asdict(self).items() if k != "password_hash"}


def _mongo():
    s = get_settings()
    if not getattr(s, "mongo_connection_string", ""):
        return None
    try:
        from pymongo import MongoClient
        client = MongoClient(s.mongo_connection_string,
                             serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client[s.mongo_database]
    except Exception as exc:
        log.warning("MongoDB unavailable (%s); using file storage.", exc)
        return None


class UserStore:
    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "users.json"
        db = _mongo()
        self.collection = (db[get_settings().mongo_users_collection]
                           if db is not None else None)
        self._seed()

    def _seed(self) -> None:
        """Create the first administrator if there is none.

        Without this a fresh deployment has no way in, since every route needs a
        principal. The password is a known default and is meant to be changed.
        """
        if self.list():
            return
        now = datetime.now(timezone.utc).isoformat()
        for username, password, name in (
            ("raushanpandey@kpmg.com", "abc123", "Raushan Pandey"),
            ("admin", "admin123", "Administrator"),
        ):
            self.save(User(username=username, email=username,
                           password_hash=hash_password(password),
                           display_name=name, role="admin",
                           domains=[ALL_DOMAINS], created_at=now))
        log.warning("Seeded default administrator accounts. Change the "
                    "passwords before this is reachable by anyone else.")

    def _read_file(self) -> list[User]:
        if not self.path.exists():
            return []
        try:
            return [User(**{k: v for k, v in row.items()
                            if k in User.__dataclass_fields__})
                    for row in json.loads(self.path.read_text(encoding="utf-8"))]
        except (json.JSONDecodeError, TypeError) as exc:
            log.error("users.json could not be read: %s", exc)
            return []

    def _write_file(self, users: list[User]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(u) for u in users], indent=2),
                             encoding="utf-8")

    def list(self) -> list[User]:
        if self.collection is not None:
            out = []
            for row in self.collection.find({}):
                row.pop("_id", None)
                try:
                    out.append(User(**{k: v for k, v in row.items()
                                       if k in User.__dataclass_fields__}))
                except TypeError:
                    continue
            return out
        return self._read_file()

    def find(self, username: str) -> Optional[User]:
        return next((u for u in self.list() if u.username == username), None)

    def save(self, user: User) -> User:
        if self.collection is not None:
            self.collection.update_one({"username": user.username},
                                       {"$set": asdict(user)}, upsert=True)
        else:
            rest = [u for u in self._read_file() if u.username != user.username]
            self._write_file(rest + [user])
        return user

    def delete(self, username: str) -> None:
        if self.collection is not None:
            self.collection.delete_one({"username": username})
        else:
            self._write_file([u for u in self._read_file()
                              if u.username != username])

    def authenticate(self, username: str, password: str) -> Optional[User]:
        user = self.find(username)
        if not user or not user.is_active:
            return None
        if not verify_password(password, user.password_hash):
            return None
        user.last_login = datetime.now(timezone.utc).isoformat()
        self.save(user)
        return user


def get_user_store() -> UserStore:
    return UserStore(get_settings().data_dir)


# ------------------------------------------------------------------ audit

@dataclass
class AuditEvent:
    timestamp: str
    user: str
    action: str
    path: str = ""
    resource: str = ""
    detail: str = ""
    ip: str = ""


class AuditLog:
    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "audit.jsonl"
        db = _mongo()
        self.collection = (db[get_settings().mongo_audit_collection]
                           if db is not None else None)

    def record(self, event: AuditEvent) -> None:
        # Auditing must never be the reason a request fails.
        try:
            if self.collection is not None:
                self.collection.insert_one(asdict(event))
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(event)) + "\n")
        except Exception as exc:
            log.warning("Audit write failed: %s", exc)

    def recent(self, limit: int = 100, user: str | None = None) -> list[dict]:
        if self.collection is not None:
            query = {"user": user} if user else {}
            rows = self.collection.find(query).sort("timestamp", -1).limit(limit)
            out = []
            for row in rows:
                row.pop("_id", None)
                out.append(row)
            return out

        if not self.path.exists():
            return []
        out = []
        for line in reversed(self.path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if user and row.get("user") != user:
                continue
            out.append(row)
            if len(out) >= limit:
                break
        return out


def get_audit_log() -> AuditLog:
    return AuditLog(get_settings().data_dir)


# ------------------------------------------------------------------ dependency

def principal_from_token(token: str) -> Principal:
    payload = decode_token(token)
    username = payload.get("sub")
    if not username:
        raise ValueError("Token carries no subject.")
    user = get_user_store().find(username)
    if user is None or not user.is_active:
        raise ValueError("That account no longer has access.")
    return Principal(key=f"jwt:{username}", name=user.display_name or username,
                     role=Role(user.role), domains=user.domains)


async def current_user(
    request: Request,
    authorization: str | None = Header(default=None),
) -> Principal:
    """Signed-in user only. Routes that must not accept an API key use this."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    try:
        return principal_from_token(authorization[7:])
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
