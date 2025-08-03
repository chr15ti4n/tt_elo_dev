# region Imports
import streamlit as st
from supabase import create_client, Client
import pandas as pd
import bcrypt
# region PIN Hashing
def hash_pin(pin: str) -> str:
    """Generates a bcrypt hash for the provided PIN."""
    return bcrypt.hashpw(pin.encode(), bcrypt.gensalt()).decode()

def check_pin(pin: str, stored: str) -> bool:
    """
    Compares the entered PIN with the stored value.
    Supports both legacy plaintext and bcrypt hashes.
    """
    if stored.startswith("$2b$") or stored.startswith("$2a$"):
        return bcrypt.checkpw(pin.encode(), stored.encode())
    else:
        return pin == stored
# endregion
# endregion

# region Supabase Setup
try:
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
except KeyError:
    st.error("Bitte setze unter [supabase] url und key in deinen Streamlit Secrets.")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
# endregion

# region Persistent Login via Query Params
q = st.query_params
def _qp_first(val):
    if isinstance(val, list):
        return val[0] if val else None
    return val
if "user" not in st.session_state:
    auto_user = _qp_first(q.get("user"))
    auto_token = _qp_first(q.get("token"))
    if auto_user and auto_token:
        # Fetch stored hash for auto_user
        resp = supabase.table("players").select("pin").eq("name", auto_user).single().execute()
        stored_hash = resp.data.get("pin") if resp.data else None
        if stored_hash and check_pin(auto_token, stored_hash):
            st.session_state.user = auto_user
# endregion

# region Authentication & CSV Display
if 'user' not in st.session_state:
    st.header("🔐 Login oder Registrierung")
    login_tab, register_tab = st.tabs(["Login", "Registrieren"])

    with login_tab:
        with st.form("login_form"):
            name = st.text_input("Name")
            pin  = st.text_input("PIN", type="password")
            if st.form_submit_button("Login"):
                resp = supabase.table("players").select("pin").eq("name", name).single().execute()
                stored_hash = resp.data.get("pin") if resp.data else None
                if stored_hash and check_pin(pin, stored_hash):
                    # Successful login: persist via URL token
                    st.session_state.user = name
                    st.query_params.update({"user": name, "token": stored_hash})
                    st.success(f"Eingeloggt als {name}")
                    st.rerun()
                else:
                    st.error("Ungültiger Name oder PIN")

    with register_tab:
        with st.form("register_form"):
            name = st.text_input("Neuer Benutzername")
            pin = st.text_input("PIN", type="password")
            pin_confirm = st.text_input("PIN wiederholen", type="password")
            if st.form_submit_button("Registrieren"):
                if not name:
                    st.error("Bitte einen Benutzernamen eingeben.")
                elif not pin:
                    st.error("Bitte eine PIN eingeben.")
                elif pin != pin_confirm:
                    st.error("PIN stimmt nicht überein.")
                else:
                    hashed = hash_pin(pin)
                    supabase.table("players").insert({"name": name, "pin": hashed}).execute()
                    st.success("Registrierung erfolgreich. Bitte einloggen.")
    st.stop()
else:
    st.header(f"👋 Willkommen, {st.session_state.user}!")
    # Show the full players CSV from Supabase
    data = supabase.table("players").select("*").execute()
    df = pd.DataFrame(data.data)
    st.dataframe(df)
# endregion
