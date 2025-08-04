# region import
from __future__ import annotations
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = object  # type: ignore
# endregion

# region supabase
TZ = ZoneInfo("Europe/Berlin")

@st.cache_resource
def get_supabase():
    """Erzeugt den Supabase‑Client aus den Streamlit‑Secrets.
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
# endregion

# region streamlit
st.set_page_config(page_title="tt-elo – Schritt 1", page_icon="🏓", layout="wide")
# endregion

# region helper
@st.cache_data(ttl=30)
def list_csv_files(folder: str) -> List[Path]:
    p = Path(folder).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        return []
    return sorted([f for f in p.glob("**/*.csv") if f.is_file()])

@st.cache_data(ttl=0)
def read_csv_full(path: Path, autodetect_sep: bool = True, tz: ZoneInfo = TZ) -> pd.DataFrame:
    """Liest eine CSV vollständig ein. Versucht automatisch das Trennzeichen zu erkennen.
    Konvertiert eine Spalte 'datum' (falls vorhanden) nach Europe/Berlin.
    """
    if autodetect_sep:
        # pandas engine='python' erkennt ; , \t automatisch, wenn sep=None
        df = pd.read_csv(path, sep=None, engine='python')
    else:
        df = pd.read_csv(path)
    # Normalize columns to lowercase for consistency
    df.columns = [str(c).lower() for c in df.columns]
    if "datum" in df.columns:
        df["datum"] = pd.to_datetime(df["datum"], errors="coerce", utc=True).dt.tz_convert(tz)
    return df

@st.cache_data(ttl=0)
def read_uploaded_csv(file, autodetect_sep: bool = True, tz: ZoneInfo = TZ) -> pd.DataFrame:
    if autodetect_sep:
        df = pd.read_csv(file, sep=None, engine='python')
    else:
        df = pd.read_csv(file)
    df.columns = [str(c).lower() for c in df.columns]
    if "datum" in df.columns:
        df["datum"] = pd.to_datetime(df["datum"], errors="coerce", utc=True).dt.tz_convert(tz)
    return df
# endregion

# region elo
# (kommt in späteren Schritten)
# endregion

# region login
# (kommt in späteren Schritten)
# endregion

# region ansicht
# region übersicht
st.title("🏓 tt-elo – Setup (Schritt 1)")

# Supabase‑Status anzeigen
sp = get_supabase()
if sp is None:
    st.info("Supabase‑Client noch nicht initialisiert. Prüfe, ob `supabase` installiert ist und Secrets gesetzt sind.")
    with st.expander("Hinweis: Secrets konfigurieren", expanded=False):
        st.code('[supabase]\nurl = "https://<PROJECT>.supabase.co"\nkey = "<ANON-ODER-SERVICE-KEY>"', language="toml")
else:
    st.success("Supabase‑Client initialisiert.")

st.markdown("""
**Ziel dieses Schritts**
1) Supabase + Streamlit eingebunden.
2) CSV‑Viewer, der komplette CSVs einliest (nicht nur einen Ausschnitt).
3) Optionaler Ordner‑Scan, um mehrere CSVs im Projekt anzuzeigen.
""")

st.header("CSV‑Viewer")
mode = st.radio("Quelle wählen", ["Ordner lesen", "Dateien hochladen"], horizontal=True)

if mode == "Ordner lesen":
    default_candidates = ["./data", "./seed_templates", "."]
    existing_defaults = [d for d in default_candidates if Path(d).exists()]
    default_folder = existing_defaults[0] if existing_defaults else "."
    folder = st.text_input("Ordnerpfad", value=str(default_folder), help="Relativ zum Projektordner.")
    if st.button("CSV‑Dateien laden", type="primary"):
        files = list_csv_files(folder)
        if not files:
            st.warning("Keine CSVs gefunden.")
        else:
            tabs = st.tabs([f.name for f in files])
            for i, f in enumerate(files):
                with tabs[i]:
                    df = read_csv_full(f)
                    st.caption(f"{f} – {len(df)} Zeilen × {len(df.columns)} Spalten")
                    st.dataframe(df, use_container_width=True)
else:
    st.write("Mehrere CSV‑Dateien hier ablegen:")
    uploads = st.file_uploader("CSV(s) auswählen)", type=["csv"], accept_multiple_files=True)
    if uploads:
        tabs = st.tabs([u.name for u in uploads])
        for i, u in enumerate(uploads):
            with tabs[i]:
                df = read_uploaded_csv(u)
                st.caption(f"{u.name} – {len(df)} Zeilen × {len(df.columns)} Spalten")
                st.dataframe(df, use_container_width=True)

st.divider()
st.markdown("""
Weiter mit **Schritt 2**: PIN‑Login & Registrierung (bcrypt‑Upgrade für Altdaten) sowie Mapping **ID↔Name**. Sag einfach *Weiter*, wenn wir das einbauen sollen.
""")
# endregion
# endregion

# region spielen
# (kommt später)
# endregion

# region account
# (kommt später)
# endregion
