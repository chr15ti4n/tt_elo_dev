# region imports
from __future__ import annotations
import streamlit as st
import pandas as pd
from zoneinfo import ZoneInfo
from typing import Optional
import bcrypt
import uuid
from supabase import create_client
# endregion

# region app_setup
st.set_page_config(page_title="AK-Tischtennis", page_icon="🏓", layout="wide")
TZ = ZoneInfo("Europe/Berlin")
# endregion

# region supabase
@st.cache_resource
def get_supabase():
    """Erzeugt den Supabase-Client aus den Streamlit-Secrets.
    Schlägt mit einer Exception fehl, wenn Secrets fehlen oder das Paket nicht installiert ist.
    """
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

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

def norm_name(s: str) -> str:
    """Normalisiert Namen für Vergleiche: entfernt alle Whitespaces und macht lowercase."""
    if s is None:
        return ""
    return "".join(str(s).split()).lower()

def find_player_by_name_normalized(name: str):
    """Sucht einen Spieler, wobei Groß/Kleinschreibung und Leerzeichen ignoriert werden."""
    target = norm_name(name)
    try:
        res = sp.table("players").select("id,name,pin,auto_token").execute()
        for rec in (res.data or []):
            if norm_name(rec.get("name", "")) == target:
                return rec
    except Exception:
        pass
    return None

def try_auto_login_from_query():
    """Auto-Login via URL-Parameter ?user=&token= (wenn players.auto_token vorhanden).
    Greift nur, wenn noch nicht eingeloggt.
    """
    if st.session_state.get("logged_in"):
        return
    q = st.query_params
    if "user" not in q or "token" not in q:
        return
    user = q.get("user")
    token = q.get("token")
    rec = find_player_by_name_normalized(user)
    if not rec:
        return
    if str(rec.get("auto_token", "")) == str(token):
        st.session_state.logged_in = True
        st.session_state.player_id = rec.get("id")
        st.session_state.player_name = rec.get("name")
        # Entferne sensible Query-Parameter aus der URL und lade sauber neu
        try:
            st.query_params.clear()
            st.rerun()
        except Exception:
            pass

# endregion

# region player_helpers

def get_current_user() -> dict | None:
    """Lädt den aktuell eingeloggten Spieler-Datensatz frisch aus Supabase."""
    try:
        pid = st.session_state.get("player_id")
        if not pid:
            return None
        rec = sp.table("players").select("*").eq("id", pid).single().execute().data
        return rec
    except Exception:
        return None

# endregion

# region login_ui
def login_register_ui():
    """UI für Login & Registrierung (PIN-basiert) inkl. "Angemeldet bleiben".
    Bei erstem Login aus Legacy-Klartext wird die PIN auf bcrypt migriert.
    """
    tabs = st.tabs(["Einloggen", "Registrieren"])    

    with tabs[0]:
        name = st.text_input("Spielername", key="login_name")
        pin = st.text_input("PIN", type="password", key="login_pin")
        remember = st.checkbox("Angemeldet bleiben")
        if st.button("Einloggen", type="primary"):
            # Spieler holen
            rec = find_player_by_name_normalized(name)
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
            # Kollisionen ignorieren Groß/Kleinschreibung und Leerzeichen
            try:
                rows = sp.table("players").select("id,name").execute().data or []
            except Exception:
                rows = []
            if any(norm_name(row.get("name")) == norm_name(r_name) for row in rows):
                st.warning("Spieler existiert bereits (Groß/Kleinschreibung/Leerzeichen ignoriert).")
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
st.title("🏓 AK-Tischtennis")
# endregion

# region session_init
# Session-Defaults setzen und Auto-Login via URL prüfen
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.player_id = None
    st.session_state.player_name = None

# Versuch: Auto-Login direkt beim Start
try_auto_login_from_query()
# endregion

# region ui_logged_in

def _metric_val(user: dict, key: str, default: int = 1200) -> int:
    try:
        return int(user.get(key, default))
    except Exception:
        return default

def logged_in_header(user: dict):
    left, right = st.columns([3,1])
    with left:
        st.markdown(f"### 👤 {user.get('name', 'Unbekannt')}")
        st.caption("Eingeloggt – Übersicht")
    with right:
        st.empty()  # Platzhalter – Logout liegt im Tab 'Account'

    cols = st.columns(4)
    cols[0].metric("G‑ELO", _metric_val(user, "g_elo"))
    cols[1].metric("Einzel", _metric_val(user, "elo"))
    cols[2].metric("Doppel", _metric_val(user, "d_elo"))
    cols[3].metric("Rundlauf", _metric_val(user, "r_elo"))


def get_current_user() -> dict | None:
    try:
        pid = st.session_state.get("player_id")
        if not pid:
            return None
        rec = sp.table("players").select("*").eq("id", pid).single().execute().data
        return rec
    except Exception:
        return None


def logged_in_ui():
    user = get_current_user()
    if not user:
        st.error("Nutzer nicht gefunden – bitte erneut einloggen.")
        for k in ("logged_in","player_id","player_name"):
            st.session_state.pop(k, None)
        st.rerun()
        return

    # Tabs: Übersicht (zuerst), Spielen (leer), Account (Logout)
    tabs = st.tabs(["Übersicht", "Spielen", "Account"])  

    # Übersicht
    with tabs[0]:
        logged_in_header(user)
        st.markdown("""
        **Übersicht**
        
        Hier bauen wir als Nächstes die Inhalte auf (z. B. letzte Spiele, Win‑Streak, offene Bestätigungen).
        """)

    # Spielen – vorerst leer (wird später implementiert)
    with tabs[1]:
        st.subheader("Spielen")
        st.info("Hier werden später Spiele erstellt (Einzel/Doppel/Rundlauf).")

    # Account – mit Logout‑Button
    with tabs[2]:
        st.subheader("Account")
        st.write(f"Angemeldet als **{user.get('name','Unbekannt')}**")
        if st.button("Logout", key="btn_logout_account", type="primary"):
            try:
                sp.table("players").update({"auto_token": None}).eq("id", user.get("id")).execute()
            except Exception:
                pass
            for k in ("logged_in","player_id","player_name"):
                st.session_state.pop(k, None)
            st.query_params.clear()
            st.rerun()

# endregion

# region auth_section
# Login/Registrierung als eigenes Fenster. Wenn nicht eingeloggt, zeigen wir NUR dieses Fenster
# und stoppen danach das Rendering der restlichen Seite.
if not st.session_state.get("logged_in"):
    st.markdown(
        """
        <style>
        .auth-card { max-width: 520px; margin: 6vh auto; padding: 1.25rem 1.25rem 0.5rem; border: 1px solid rgba(255,255,255,0.15); border-radius: 12px; background: rgba(0,0,0,0.25); box-shadow: 0 8px 28px rgba(0,0,0,.35); }
        .auth-title { text-align:center; font-size: 1.1rem; margin: 0.25rem 0 0.75rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="auth-card"><div class="auth-title">🔐 Login & Registrierung</div>', unsafe_allow_html=True)
    login_register_ui()
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()
else:
    logged_in_ui()
# endregion
