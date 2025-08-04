# region imports
from __future__ import annotations
import streamlit as st
import pandas as pd
from zoneinfo import ZoneInfo
from typing import Optional
import bcrypt
import uuid
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = object  # type: ignore
# endregion

# region app_setup
st.set_page_config(page_title="AK-Tischtennis", page_icon="🏓", layout="wide")
TZ = ZoneInfo("Europe/Berlin")
# endregion

# region supabase
@st.cache_resource
def get_supabase():
    """Erzeugt den Supabase-Client aus den Streamlit-Secrets.
    Gibt None zurück, wenn Secrets fehlen oder das Paket nicht installiert ist.
    """
    if create_client is None:
        return None
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
    except Exception:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None

sp = get_supabase()
# endregion

# region data_loading
@st.cache_data(ttl=30)
def load_table(table_name: str) -> pd.DataFrame:
    """Lädt eine Supabase-Tabelle vollständig in ein DataFrame.
    - Spaltennamen -> lower()
    - 'datum' -> Europe/Berlin
    - Bei Fehler (z. B. Tabelle existiert nicht) leeres DF
    """
    if sp is None:
        return pd.DataFrame()
    try:
        res = sp.table(table_name).select("*").execute()
        data = res.data or []
    except Exception:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df.columns = [str(c).lower() for c in df.columns]
    if "datum" in df.columns:
        df["datum"] = pd.to_datetime(df["datum"], errors="coerce", utc=True).dt.tz_convert(TZ)
    return df

def clear_table_cache():
    load_table.clear()
# endregion

# region auth_helpers
def hash_pin(pin: str) -> str:
    return bcrypt.hashpw(pin.encode(), bcrypt.gensalt()).decode()

def check_pin(entered: str, stored: str) -> bool:
    """Vergleicht eingegebene PIN mit gespeichertem Wert (unterstützt Legacy-Klartext)."""
    if stored and (stored.startswith("$2b$") or stored.startswith("$2a$")):
        try:
            return bcrypt.checkpw(entered.encode(), stored.encode())
        except Exception:
            return False
    return entered == (stored or "")

def try_auto_login_from_query():
    """Auto-Login via URL-Parameter ?user=&token= (wenn players.auto_token vorhanden).
    Greift nur, wenn noch nicht eingeloggt.
    """
    if sp is None or st.session_state.get("logged_in"):
        return
    q = st.query_params
    if "user" not in q or "token" not in q:
        return
    user = q.get("user")
    token = q.get("token")
    try:
        rec = sp.table("players").select("*").eq("name", user).single().execute().data
    except Exception:
        rec = None
    if not rec:
        return
    if str(rec.get("auto_token", "")) == str(token):
        st.session_state.logged_in = True
        st.session_state.player_id = rec.get("id")
        st.session_state.player_name = rec.get("name")

# endregion

# region login_ui
def login_register_ui():
    """UI für Login & Registrierung (PIN-basiert) inkl. "Angemeldet bleiben".
    Bei erstem Login aus Legacy-Klartext wird die PIN auf bcrypt migriert.
    """
    if sp is None:
        st.error("Supabase-Client nicht initialisiert.")
        with st.expander("Secrets konfigurieren", expanded=False):
            st.code('[supabase]\nurl = "https://<PROJECT>.supabase.co"\nkey = "<ANON-ODER-SERVICE-KEY>"', language="toml")
        return

    tabs = st.tabs(["Einloggen", "Registrieren"])    

    with tabs[0]:
        name = st.text_input("Spielername", key="login_name")
        pin = st.text_input("PIN", type="password", key="login_pin")
        remember = st.checkbox("Angemeldet bleiben")
        if st.button("Einloggen", type="primary"):
            # Spieler holen
            try:
                rec = sp.table("players").select("*").eq("name", name).single().execute().data
            except Exception:
                rec = None
            if not rec:
                st.error("Spielername nicht gefunden.")
                return
            stored_pin = str(rec.get("pin", ""))
            if not check_pin(pin, stored_pin):
                st.error("PIN falsch.")
                return
            # ggf. Legacy -> bcrypt upgraden
            if not (stored_pin.startswith("$2b$") or stored_pin.startswith("$2a$")):
                try:
                    sp.table("players").update({"pin": hash_pin(pin)}).eq("id", rec["id"]).execute()
                except Exception:
                    pass
            # Session setzen
            st.session_state.logged_in = True
            st.session_state.player_id = rec.get("id")
            st.session_state.player_name = rec.get("name")

            # Remember-Me via auto_token + URL-Params
            if remember:
                try:
                    token = uuid.uuid4().hex
                    # Versuch: Spalte auto_token aktualisieren (falls nicht vorhanden, wird es fehlschlagen)
                    sp.table("players").update({"auto_token": token}).eq("id", rec["id"]).execute()
                    st.query_params.update({"user": rec.get("name"), "token": token})
                except Exception:
                    st.info("Hinweis: 'auto_token' Spalte nicht vorhanden – automatischer Login per URL-Token ist deaktiviert.")
            st.success(f"Willkommen, {rec.get('name')}!")
            st.rerun()

    with tabs[1]:
        r_name = st.text_input("Neuer Spielername", key="reg_name")
        r_pin1 = st.text_input("PIN wählen (4-stellig)", type="password", key="reg_pin1")
        r_pin2 = st.text_input("PIN bestätigen", type="password", key="reg_pin2")
        if st.button("Registrieren"):
            if not r_name or not r_pin1 or not r_pin2:
                st.warning("Name und PIN eingeben.")
                return
            if r_pin1 != r_pin2:
                st.warning("PINs stimmen nicht überein.")
                return
            # existiert schon?
            try:
                exists = sp.table("players").select("id").eq("name", r_name).maybe_single().execute().data
            except Exception:
                exists = None
            if exists:
                st.warning("Spieler existiert bereits.")
                return
            payload = {
                "name": r_name,
                "pin": hash_pin(r_pin1),
                # Stats-Defaults, falls Spalten existieren
                "elo": 1200, "siege": 0, "niederlagen": 0, "spiele": 0,
                "d_elo": 1200, "d_siege": 0, "d_niederlagen": 0, "d_spiele": 0,
                "r_elo": 1200, "r_siege": 0, "r_zweite": 0, "r_niederlagen": 0, "r_spiele": 0,
                "g_elo": 1200,
            }
            try:
                sp.table("players").insert(payload).execute()
                clear_table_cache()
                st.success("Registriert! Bitte einloggen.")
            except Exception as e:
                st.error(f"Konnte nicht registrieren: {e}")
# endregion

# region layout_header
st.title("🏓 tt-elo – Supabase Datenbrowser")

# endregion

# region auth_section
st.divider()
st.subheader("Login & Registrierung")
if not st.session_state.get("logged_in"):
    login_register_ui()
else:
    st.success("Bereit – du bist eingeloggt. Als nächstes können wir Eintragen/Bestätigen bauen.")
# endregion
