from __future__ import annotations
import streamlit as st
import pandas as pd
from zoneinfo import ZoneInfo
from typing import Optional
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = object  # type: ignore

# ---------- App Setup ----------
st.set_page_config(page_title="tt-elo – Datenbrowser", page_icon="🏓", layout="wide")
TZ = ZoneInfo("Europe/Berlin")

# ---------- Supabase ----------
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

# ---------- Daten laden ----------
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

# ---------- UI ----------
st.title("🏓 tt-elo – Supabase Datenbrowser")

if sp is None:
    st.error("Supabase-Client nicht initialisiert.")
    with st.expander("Secrets konfigurieren", expanded=False):
        st.code('[supabase]\nurl = "https://<PROJECT>.supabase.co"\nkey = "<ANON-ODER-SERVICE-KEY>"', language="toml")
    st.stop()
else:
    st.success("Supabase-Client initialisiert.")

# Refresh-Button
cols = st.columns([1,3])
if cols[0].button("Aktualisieren", type="primary"):
    load_table.clear()
    st.experimental_rerun()

st.caption("Die folgenden Tabellen werden direkt aus Supabase geladen (select *).")

tables = [
    "players",
    "matches",
    "doubles",
    "rounds",
    "pending_matches",
    "pending_doubles",
    "pending_rounds",
]

for t in tables:
    with st.expander(f"{t}", expanded=False):
        df = load_table(t)
        if df.empty:
            st.info("Keine Daten vorhanden oder Tabelle (noch) nicht angelegt.")
        else:
            st.caption(f"{len(df)} Zeilen × {len(df.columns)} Spalten")
            st.dataframe(df, use_container_width=True)

st.divider()
st.markdown("Das ist nur der Daten-Browser. Als nächstes bauen wir PIN-Login & Eintragen/Bestätigen auf diesem Layer auf.")
