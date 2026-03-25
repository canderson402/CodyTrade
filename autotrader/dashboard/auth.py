"""Authentication for the dashboard — passwords loaded from .env, never in code.

To change a password:
  1. Generate a hash:  python -c "import bcrypt; print(bcrypt.hashpw(b'newpass', bcrypt.gensalt()).decode())"
  2. Paste it into .env as ADMIN_PASSWORD_HASH or VIEWER_PASSWORD_HASH

Roles:
  admin  — full access: view everything, change settings, trigger actions
  viewer — read-only: can see dashboards and data, cannot change anything
"""

import os
import bcrypt
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """Read a secret from os.environ (local) or st.secrets (Streamlit Cloud)."""
    value = os.getenv(key, "")
    if not value:
        try:
            value = st.secrets.get(key, default)
        except Exception:
            value = default
    return value


# Password hashes loaded from .env or Streamlit secrets — never hardcoded
_ADMIN_HASH: str = _get_secret("ADMIN_PASSWORD_HASH")
_VIEWER_HASH: str = _get_secret("VIEWER_PASSWORD_HASH")

USERS: dict[str, dict] = {
    "admin": {"hash": _ADMIN_HASH, "role": "admin"},
    "viewer": {"hash": _VIEWER_HASH, "role": "viewer"},
}


def _verify_password(username: str, password: str) -> bool:
    """Check a plaintext password against the stored bcrypt hash."""
    user = USERS.get(username)
    if user is None or not user["hash"]:
        return False
    try:
        return bcrypt.checkpw(password.encode(), user["hash"].encode())
    except Exception:
        return False


def login_gate() -> bool:
    """Show login form if not authenticated. Returns True if user is logged in."""
    if st.session_state.get("authenticated"):
        return True

    st.title("AutoTrader Login")

    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Log in", type="primary"):
        if _verify_password(username, password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["role"] = USERS[username]["role"]
            st.rerun()
        else:
            st.error("Invalid username or password.")

    return False


def logout() -> None:
    """Clear authentication state."""
    st.session_state["authenticated"] = False
    st.session_state["username"] = ""
    st.session_state["role"] = ""
    st.rerun()


def is_admin() -> bool:
    """Check if the current user has admin privileges."""
    return st.session_state.get("role") == "admin"


def get_username() -> str:
    """Get the current logged-in username."""
    return st.session_state.get("username", "")
