# region Imports
import streamlit as st
from contextlib import contextmanager
import numpy as np

@contextmanager
def ui_container(title: str):
    """Use st.modal if available, else st.expander fallback."""
    if hasattr(st, "modal"):
        with st.modal(title):
            yield
    else:
        with st.expander(title, expanded=True):
            yield
            
import pandas as pd
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo 
import bcrypt
# QR-Code generation
import qrcode
# Google Sheets
import os
import functools
import time
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe

# region Paths
# ---------- Konstante Pfade ----------
PLAYERS = Path("players.csv")
MATCHES = Path("matches.csv")
PENDING = Path("pending_matches.csv")
PENDING_D = Path("pending_doubles.csv")
DOUBLES   = Path("doubles.csv")
PENDING_R = Path("pending_rounds.csv")  
ROUNDS    = Path("rounds.csv")          
# endregion


# ---------- QR-Code für Schnellzugriff ----------
QR_FILE = Path("form_qr.png")
APP_URL  = "https://tt-elo.streamlit.app"
if not QR_FILE.exists():
    qr_img = qrcode.make(APP_URL)
    qr_img.save(QR_FILE)

# ---------- Google Sheets ----------
USE_GSHEETS = "gcp" in st.secrets  # Nur aktiv, wenn Service‑Account‑Creds hinterlegt

# Google Sheets Caching/Singleton helpers
if USE_GSHEETS:

    @st.cache_resource
    def _get_sheet():
        gc_local = gspread.service_account_from_dict(st.secrets["gcp"])
        spread_id = st.secrets.get("spread_id", "")
        if spread_id:
            return gc_local.open_by_key(spread_id)
        return gc_local.open("tt_elo")

    sh = _get_sheet()  # einmal pro Session

    SHEET_NAMES = {
        "players.csv":  "players",
        "matches.csv":  "matches",
        "pending_matches.csv": "pending_matches",
        "pending_doubles.csv": "pending_doubles",
        "doubles.csv":  "doubles",
        "pending_rounds.csv": "pending_rounds",
        "rounds.csv":   "rounds",
    }

    @functools.lru_cache(maxsize=None)
    def _get_ws(name: str, cols: list[str]):
        """Gibt Worksheet‑Objekt; legt es bei Bedarf an (cached)."""
        try:
            return sh.worksheet(name)
        except gspread.WorksheetNotFound:
            return sh.add_worksheet(name, rows=1000, cols=len(cols))
# endregion


# region Helper Functions
# ---------- Hilfsfunktionen ----------

# --------- Session‑Cache für DataFrames (verhindert unnötige Sheets‑Reads) ---------
if "dfs" not in st.session_state:   # Key: Path.name  |  Value: DataFrame
    st.session_state["dfs"] = {}

def save_csv(df: pd.DataFrame, path: Path):
    """Speichert DataFrame entweder als lokale CSV oder in Google‑Sheets."""
    if USE_GSHEETS and path.name in SHEET_NAMES:
        ws_name = SHEET_NAMES[path.name]
        ws = _get_ws(ws_name, tuple(df.columns))
        # gspread verträgt keine Datetime‑Objekte → zuerst in Strings wandeln
        df_to_write = df.copy()
        for col in df_to_write.select_dtypes(include=["datetimetz", "datetime64[ns, UTC]", "datetime64[ns]"]).columns:
            df_to_write[col] = df_to_write[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        # Google Sheets kann kein NaN serialisieren → NaN durch leere Strings ersetzen
        df_to_write = df_to_write.fillna("")
        ws.clear()
        set_with_dataframe(ws, df_to_write.reset_index(drop=True))
        # Cache aktualisieren
        st.session_state["dfs"][path.name] = df.copy()
        time.sleep(0.1)  # Throttle to avoid hitting per‑minute quota
    else:
        df.to_csv(path, index=False)
        # Cache aktualisieren
        st.session_state["dfs"][path.name] = df.copy()


def load_or_create(path: Path, cols: list[str]) -> pd.DataFrame:
    """Lädt DataFrame aus CSV oder Google‑Sheets; legt bei Bedarf leere Tabelle an."""
    # Zuerst Session‑Cache prüfen
    if path.name in st.session_state["dfs"]:
        return st.session_state["dfs"][path.name]

    if USE_GSHEETS and path.name in SHEET_NAMES:
        ws = _get_ws(SHEET_NAMES[path.name], tuple(cols))
        df = get_as_dataframe(ws).dropna(how="all")
        # Falls Sheet gerade frisch angelegt → Kopfzeile schreiben
        if df.empty and ws.row_count == 0:
            set_with_dataframe(ws, pd.DataFrame(columns=cols))
            df = pd.DataFrame(columns=cols)
            st.session_state["dfs"][path.name] = df.copy()
            return df
        result_df = df if not df.empty else pd.DataFrame(columns=cols)
        st.session_state["dfs"][path.name] = result_df.copy()
        return result_df
    else:
        if path.exists():
            df = pd.read_csv(path)
            st.session_state["dfs"][path.name] = df.copy()
            return df
        df = pd.DataFrame(columns=cols)
        st.session_state["dfs"][path.name] = df.copy()
        return df

def calc_elo(r_a, r_b, score_a, k=32):
    """ELO‑Formel mit Punktdifferenz.
    score_a ∈ [0,1]   1 = 11:0   0.09 ≈ 11:10   0 = Niederlage"""
    exp_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    return round(r_a + k * (score_a - exp_a), 0)

# --------- Gesamt-ELO berechnen -----------
def compute_gelo(df: pd.DataFrame,
                 w_e=0.6, w_d=0.25, w_r=0.15) -> pd.DataFrame:
    df["G_ELO"] = (
        w_e * df["ELO"] +
        w_d * df["D_ELO"] +
        w_r * df["R_ELO"]
    ).round(0).astype(int)
    return df
# endregion

# region PIN Hashing
# ---------- PIN-Hashing ----------
def hash_pin(pin: str) -> str:
    return bcrypt.hashpw(pin.encode(), bcrypt.gensalt()).decode()

def check_pin(pin: str, stored: str) -> bool:
    """
    Vergleicht die eingegebene PIN mit dem gespeicherten Wert.
    Unterstützt sowohl Klartext (Legacy) als auch bcrypt-Hashes.
    """
    if stored.startswith("$2b$") or stored.startswith("$2a$"):
        # bcrypt-Hash
        return bcrypt.checkpw(pin.encode(), stored.encode())
    else:
        # Legacy: Klartext
        return pin == stored
# endregion

# region Rebuild Players (Einzel)
# ---------- Spieler-Stats & ELO komplett neu berechnen ----------
def rebuild_players(players_df: pd.DataFrame, matches_df: pd.DataFrame, k: int = 64) -> pd.DataFrame:
    """
    Setzt alle Spieler-Statistiken zurück und berechnet sie anhand der
    chronologisch sortierten Match-Liste neu.
    """
    players_df = players_df.copy()
    # Basiswerte zurücksetzen
    players_df[["ELO", "Siege", "Niederlagen", "Spiele"]] = 0
    players_df["ELO"] = 1200

    if matches_df.empty:
        return players_df

    # Matches nach Datum aufsteigend sortieren
    matches_sorted = matches_df.sort_values("Datum")

    for _, row in matches_sorted.iterrows():
        a, b = row["A"], row["B"]
        pa, pb = int(row["PunkteA"]), int(row["PunkteB"])

        # Falls Spieler inzwischen gelöscht wurden, Match überspringen
        if a not in players_df["Name"].values or b not in players_df["Name"].values:
            continue

        r_a = players_df.loc[players_df["Name"] == a, "ELO"].iat[0]
        r_b = players_df.loc[players_df["Name"] == b, "ELO"].iat[0]

        if pa == pb:
            continue  # Unentschieden ignorieren
        # K‑Faktor skalieren nach Punktdifferenz
        margin  = abs(pa - pb)            # 0–11
        k_eff   = k * (1 + margin / 11)   # 0‑Siege → K, 11‑0 → 2·K
        winner_is_a = pa > pb

        new_r_a = calc_elo(r_a, r_b, 1 if winner_is_a else 0, k_eff)
        new_r_b = calc_elo(r_b, r_a, 0 if winner_is_a else 1, k_eff)

        # --- Statistiken (ganze Siege/Niederlagen, 1 Spiel) ----------------
        if pa > pb:
            win_a, win_b = 1, 0
        else:
            win_a, win_b = 0, 1

        players_df.loc[players_df["Name"] == a, ["ELO", "Siege", "Niederlagen", "Spiele"]] = [
            new_r_a,
            players_df.loc[players_df["Name"] == a, "Siege"].iat[0] + win_a,
            players_df.loc[players_df["Name"] == a, "Niederlagen"].iat[0] + win_b,
            players_df.loc[players_df["Name"] == a, "Spiele"].iat[0] + 1,
        ]
        players_df.loc[players_df["Name"] == b, ["ELO", "Siege", "Niederlagen", "Spiele"]] = [
            new_r_b,
            players_df.loc[players_df["Name"] == b, "Siege"].iat[0] + win_b,
            players_df.loc[players_df["Name"] == b, "Niederlagen"].iat[0] + win_a,
            players_df.loc[players_df["Name"] == b, "Spiele"].iat[0] + 1,
        ]
    players_df = compute_gelo(players_df)
    return players_df
# endregion

# region Doppel ELO Helper
# ---------- Doppel-ELO ----------
def calc_doppel_elo(r1, r2, opp_avg, s, k=24):
    team_avg = (r1 + r2) / 2
    exp = 1 / (1 + 10 ** ((opp_avg - team_avg) / 400))
    delta = k * (s - exp)
    return round(r1 + delta), round(r2 + delta)

# ---------- Daten laden ----------
players = load_or_create(PLAYERS, ["Name", "ELO", "Siege", "Niederlagen", "Spiele", "Pin"])
# Falls alte CSV noch keine Pin‑Spalte hatte
if "Pin" not in players.columns:
    players["Pin"] = ""
# Doppel-Spalten ergänzen
for col in ["D_ELO", "D_Siege", "D_Niederlagen", "D_Spiele"]:
    if col not in players.columns:
        players[col] = 1200 if col == "D_ELO" else 0
# Rundlauf-Spalten ergänzen
for col in ["R_ELO", "R_Siege", "R_Zweite", "R_Niederlagen", "R_Spiele"]:
    if col not in players.columns:
        players[col] = 1200 if col == "R_ELO" else 0
        
players = compute_gelo(players)
# Daten laden
matches = load_or_create(MATCHES, ["Datum", "A", "B", "PunkteA", "PunkteB"])
pending = load_or_create(PENDING, ["Datum", "A", "B", "PunkteA", "PunkteB", "confA", "confB"])
pending_d = load_or_create(PENDING_D, ["Datum","A1","A2","B1","B2","PunkteA","PunkteB","confA","confB"])
doubles   = load_or_create(DOUBLES,   ["Datum","A1","A2","B1","B2","PunkteA","PunkteB"])
pending_r = load_or_create(PENDING_R, ["Datum","Teilnehmer","Finalist1","Finalist2","Sieger","creator","confirmed_by"])
# -- Kompatibilität älterer CSV-Versionen --
if "confirmed_by" not in pending_r.columns:
    if "conf" in pending_r.columns:
        pending_r = pending_r.rename(columns={"conf": "confirmed_by"})
    else:
        pending_r["confirmed_by"] = ""
    save_csv(pending_r, PENDING_R)
rounds    = load_or_create(ROUNDS,    ["Datum","Teilnehmer","Finalist1","Finalist2","Sieger"])

# Ensure confA/confB are boolean for logical operations
pending["confA"]   = pending["confA"].astype(bool)
pending["confB"]   = pending["confB"].astype(bool)
pending_d["confA"] = pending_d["confA"].astype(bool)
pending_d["confB"] = pending_d["confB"].astype(bool)

for df in (matches, pending, pending_d, doubles, pending_r, rounds):
    if not df.empty:
        df["Datum"] = (
            pd.to_datetime(df["Datum"], utc=True, errors="coerce")
              .dt.tz_convert("Europe/Berlin")
        )
# endregion

# region Doppel ELO Rebuild
# ---------- Doppel-Stats & ELO komplett neu berechnen ----------
def rebuild_players_d(players_df, doubles_df, k=48):
    players_df = players_df.copy()
    players_df[["D_ELO","D_Siege","D_Niederlagen","D_Spiele"]] = 0
    players_df["D_ELO"] = 1200
    for _, row in doubles_df.sort_values("Datum").iterrows():
        a1,a2,b1,b2 = row[["A1","A2","B1","B2"]]
        pa,pb = int(row["PunkteA"]), int(row["PunkteB"])
        ra1 = players_df.loc[players_df.Name==a1,"D_ELO"].iat[0]
        ra2 = players_df.loc[players_df.Name==a2,"D_ELO"].iat[0]
        rb1 = players_df.loc[players_df.Name==b1,"D_ELO"].iat[0]
        rb2 = players_df.loc[players_df.Name==b2,"D_ELO"].iat[0]
        a_avg,b_avg = (ra1+ra2)/2, (rb1+rb2)/2
        if pa == pb:
            continue  # kein Unentschieden
        margin  = abs(pa - pb)
        k_eff   = k * (1 + margin / 11)
        team_a_win = 1 if pa > pb else 0

        nr1, nr2 = calc_doppel_elo(ra1, ra2, b_avg, team_a_win, k_eff)
        nr3, nr4 = calc_doppel_elo(rb1, rb2, a_avg, 1 - team_a_win, k_eff)
        updates = [
            (a1, nr1, team_a_win),
            (a2, nr2, team_a_win),
            (b1, nr3, 1 - team_a_win),
            (b2, nr4, 1 - team_a_win),
        ]
        for p,new,s in updates:
            players_df.loc[players_df.Name==p, ["D_ELO","D_Siege","D_Niederlagen","D_Spiele"]] = [
                new,
                players_df.loc[players_df.Name==p,"D_Siege"].iat[0] + s,
                players_df.loc[players_df.Name==p,"D_Niederlagen"].iat[0] + (1 - s),
                players_df.loc[players_df.Name==p,"D_Spiele"].iat[0] + 1,
            ]
    players_df = compute_gelo(players_df)
    return players_df
# endregion

# region Rundlauf ELO
# ---------- Rundlauf-ELO ----------
def calc_round_elo(r, avg, s, k=48):
    """
    r      ... Rating des Spielers
    avg    ... Durchschnittsrating aller Teilnehmer
    s      ... Ergebnis (1 Sieger, 0.5 Zweiter, 0 Verlierer)
    """
    exp = 1 / (1 + 10 ** ((avg - r) / 400))
    return round(r + k * (s - exp))

# ---------- Rundlauf-Stats & ELO komplett neu berechnen ----------
def rebuild_players_r(players_df, rounds_df, k=48):
    players_df = players_df.copy()
    players_df[["R_ELO","R_Siege","R_Zweite","R_Niederlagen","R_Spiele"]] = 0
    players_df["R_ELO"] = 1200
    if rounds_df.empty:
        return players_df
    for _, row in rounds_df.sort_values("Datum").iterrows():
        teilnehmer = row["Teilnehmer"].split(";")
        fin1, fin2, winner = row["Finalist1"], row["Finalist2"], row["Sieger"]
        avg = players_df.loc[players_df.Name.isin(teilnehmer), "R_ELO"].mean()
        deltas = {}
        for p in teilnehmer:
            old = players_df.loc[players_df.Name==p,"R_ELO"].iat[0]
            if p == winner:
                s = 1
                players_df.loc[players_df.Name==p,"R_Siege"] += 1
            elif p in (fin1, fin2):
                s = 0.5
                players_df.loc[players_df.Name==p,"R_Zweite"] += 1
            else:
                s = 0
                players_df.loc[players_df.Name==p,"R_Niederlagen"] += 1
            exp = 1 / (1 + 10 ** ((avg - old) / 400))
            delta = k * (s - exp)
            deltas[p] = delta
        # Null‑Summe: Offset so dass Summe(delta_adj) = 0
        offset = sum(deltas.values()) / len(deltas)
        for p, delta in deltas.items():
            new = round(players_df.loc[players_df.Name==p,"R_ELO"].iat[0] + (delta - offset))
            players_df.loc[players_df.Name==p,"R_ELO"] = new
            players_df.loc[players_df.Name==p,"R_Spiele"] += 1
    players_df = compute_gelo(players_df)
    return players_df
# endregion


# region Auth & Sidebar UI
# ---------- Login / Registrierung ----------
# --- Modal flags (Einzel / Doppel / Rundlauf / Bestätigen) ---
for _flag in ("show_single_modal", "show_double_modal", "show_round_modal", "show_confirm_modal"):
    if _flag not in st.session_state:
        st.session_state[_flag] = False

# -------- Modal helper: open one modal at a time ----------
def _open_modal(which: str):
    """Set exactly one modal flag True, others False."""
    for f in ("show_single_modal", "show_double_modal", "show_round_modal", "show_confirm_modal"):
        st.session_state[f] = (f == which)

# -------- Rebuild all ratings helper ----------
def _rebuild_all():
    """Rebuilds all ELO ratings after a confirmed match and saves players.csv."""
    global players, matches, doubles, rounds
    players = rebuild_players(players, matches)
    players = rebuild_players_d(players, doubles)
    players = rebuild_players_r(players, rounds)
    save_csv(players, PLAYERS)
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_player" not in st.session_state:
    st.session_state.current_player = None
# -------- Auto‑Login per URL ?user=&token= ----------
if not st.session_state.logged_in:
    q = st.query_params
    if "user" in q and "token" in q:
        auto_user  = q["user"][0]
        auto_token = q["token"][0]
        if auto_user in players["Name"].values:
            stored_pin = players.loc[players["Name"] == auto_user, "Pin"].iat[0]
            if stored_pin == auto_token:
                st.session_state.logged_in = True
                st.session_state.current_player = auto_user
# View mode: "spiel" (default) or "regeln"
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "home"

# Remove sidebar wrapper and dedent login/registration UI to main area
if not st.session_state.logged_in:
    st.header("Login / Registrieren")
    default_mode = "Registrieren" if players.empty else "Login"
    mode = st.radio("Aktion wählen", ("Login", "Registrieren"),
                    index=0 if default_mode == "Login" else 1)

    if mode == "Login":
        if players.empty:
            st.info("Noch keine Spieler angelegt.")
        else:
            login_name = st.selectbox(
                "Spieler",
                players["Name"],
                index=None,
                placeholder="Name wählen"
            )
            login_pin = st.text_input("PIN", type="password")
            if login_name is None:
                st.stop()
            if st.button("Einloggen"):
                stored_pin = players.loc[players["Name"] == login_name, "Pin"].iat[0]
                if check_pin(login_pin, stored_pin):
                    # Falls PIN noch im Klartext war: sofort hash speichern
                    if not stored_pin.startswith("$2b$") and not stored_pin.startswith("$2a$"):
                        players.loc[players["Name"] == login_name, "Pin"] = hash_pin(login_pin)
                        save_csv(players, PLAYERS)
                    st.session_state.logged_in = True
                    st.session_state.current_player = login_name
                    # Save login in URL so refresh preserves session
                    st.query_params.update({
                        "user": login_name,
                        "token": players.loc[players["Name"] == login_name, "Pin"].iat[0],
                    })
                    st.rerun()
                else:
                    st.error("Falsche PIN")

    elif mode == "Registrieren":
        reg_name = st.text_input("Neuer Spielername")
        reg_pin1 = st.text_input("PIN wählen (4-stellig)", type="password")
        reg_pin2 = st.text_input("PIN bestätigen", type="password")
        if st.button("Registrieren"):
            if reg_name == "" or reg_pin1 == "":
                st.warning("Name und PIN eingeben.")
            elif reg_pin1 != reg_pin2:
                st.warning("PINs stimmen nicht überein.")
            elif reg_name in players["Name"].values:
                st.warning("Spieler existiert bereits.")
            else:
                new_player = {
                    "Name": reg_name,
                    "ELO": 1200,
                    "Siege": 0,
                    "Niederlagen": 0,
                    "Spiele": 0,
                    "Pin": hash_pin(reg_pin1),

                    # Doppel‑Defaults
                    "D_ELO": 1200,
                    "D_Siege": 0,
                    "D_Niederlagen": 0,
                    "D_Spiele": 0,

                    # Rundlauf‑Defaults
                    "R_ELO": 1200,
                    "R_Siege": 0,
                    "R_Zweite": 0,
                    "R_Niederlagen": 0,
                    "R_Spiele": 0,
                }
                players = pd.concat([players, pd.DataFrame([new_player])], ignore_index=True)
                players = compute_gelo(players)  # Gesamt-ELO für neuen Spieler
                save_csv(players, PLAYERS)
                st.success(f"{reg_name} angelegt. Jetzt einloggen.")
                st.rerun()
# Eingeloggt: Sidebar zeigt Menü und Logout
else:
    with st.sidebar:
        current_player = st.session_state.current_player  # lokal verfügbar
        st.markdown(f"**Eingeloggt als:** {current_player}")
        # Zurück zum Dashboard
        if st.button("🏓 Home", use_container_width=True):
            _open_modal("")                    # alle Modals schließen
            st.session_state.view_mode = "home"
            st.rerun()

        if st.button("♻️ Aktualisieren", use_container_width=True):
            # Cache leeren, damit neu aus Google‑Sheets geladen wird
            if "dfs" in st.session_state:
                st.session_state["dfs"].clear()
            try:
                _get_ws.cache_clear()   # Worksheet‑Cache leeren
            except Exception:
                pass
            st.rerun()

        if st.button("📜 Regeln", use_container_width=True):
            st.session_state.view_mode = "regeln"
            st.rerun()

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.current_player = None
            st.query_params.clear()  # clear
            st.rerun()

        # QR-Code für Match-Eintrag
        with st.expander("📱 QR-Code"):
            st.image(str(QR_FILE), width=180)
            st.caption("Scanne, um zu spielen 🏓.")

        # Account löschen (Selbstlöschung)
        with st.expander("🗑️ Account löschen"):
            st.warning("Dies löscht deinen Spieler‑Eintrag **dauerhaft** inklusive aller zugehörigen Spiele!")
            confirm = st.checkbox("Ich bin mir absolut sicher.")
            if st.button("Account unwiderruflich löschen") and confirm:
                # Spieler aus players entfernen
                players = players[players["Name"] != current_player]

                # Helferfunktion: Zeilen entfernen, in denen Name auftaucht
                def drop_rows(df: pd.DataFrame, cols: list[str]):
                    if df.empty:
                        return df
                    mask = df[cols].astype(str).eq(current_player).any(axis=1)
                    return df[~mask]

                # Einzel
                matches   = drop_rows(matches,   ["A", "B"])
                pending   = drop_rows(pending,   ["A", "B"])

                # Doppel
                doubles    = drop_rows(doubles,    ["A1","A2","B1","B2"])
                pending_d  = drop_rows(pending_d,  ["A1","A2","B1","B2"])

                # Rundlauf
                if not rounds.empty:
                    rounds = rounds[~rounds["Teilnehmer"].str.contains(current_player)]
                if not pending_r.empty:
                    pending_r = pending_r[~pending_r["Teilnehmer"].str.contains(current_player)]

                # Elo neu berechnen
                players = rebuild_players(players, matches)
                players = rebuild_players_d(players, doubles)
                players = rebuild_players_r(players, rounds)

                # CSV speichern
                save_csv(players,   PLAYERS)
                save_csv(matches,   MATCHES)
                save_csv(pending,   PENDING)
                save_csv(doubles,   DOUBLES)
                save_csv(pending_d, PENDING_D)
                save_csv(rounds,    ROUNDS)
                save_csv(pending_r, PENDING_R)

                st.query_params.clear()  # clear URL params
                st.success("Account und alle zugehörigen Daten wurden gelöscht.")
                st.session_state.logged_in = False
                st.session_state.current_player = None
                st.rerun()


if not st.session_state.logged_in:
    st.stop()
#
# endregion


current_player = st.session_state.current_player

# region Home Ansicht
if st.session_state.view_mode == "home":
    st.title("🏓 Tischtennis-Dashboard")
    user = players.loc[players.Name == current_player].iloc[0]

    st.markdown(f"### Willkommen, **{current_player}**!")
    st.metric("Gesamt-ELO", int(user.G_ELO))

    cols = st.columns(3)
    cols[0].metric("Einzel",  int(user.ELO))
    cols[1].metric("Doppel",  int(user.D_ELO))
    cols[2].metric("Rundlauf", int(user.R_ELO))

    st.divider()

    def mini_lb(df: pd.DataFrame, elo_col: str, title: str, height: int = 350):
        """
        Zeigt ein Leaderboard für die angegebene Spielform.
        * Vollständige Liste (kein Head‑10‑Cut mehr)
        * Zeile des aktuellen Spielers gelb hinterlegt
        * Scrollbar (fixe Höhe)
        """
        tab = (df.sort_values(elo_col, ascending=False)
                 .loc[:, ["Name", elo_col]]
                 .rename(columns={elo_col: "ELO"})
                 .reset_index(drop=True))
        tab["ELO"] = tab["ELO"].astype(int)

        # Highlightfunktion
        def _highlight(row):
            return ['background-color: #fff3b0' if row["Name"] == current_player else '' for _ in row]

        styled = tab.style.apply(_highlight, axis=1)

        st.subheader(title)
        st.dataframe(styled, hide_index=True, width=350, height=height)
    
    # Gesamt‑ELO (alle Spieler)
    mini_lb(players, "G_ELO", "Gesamt – Ranking")

    mini_lb(players[players.Spiele   > 0], "ELO",   "Einzel – Ranking", height=175)
    mini_lb(players[players.D_Spiele > 0], "D_ELO", "Doppel – Ranking", height=175)
    mini_lb(players[players.R_Spiele > 0], "R_ELO", "Rundlauf – Ranking", height=175)

    st.divider()
    # --- Pending Confirmation Counts ------------------------------
    # Einzel
    sp = pending[
        (
            (pending["A"] == current_player) & (~pending["confA"])
        ) | (
            (pending["B"] == current_player) & (~pending["confB"])
        )
    ]
    # Doppel
    dp = pending_d[
        (
            (pending_d["A1"] == current_player) | (pending_d["A2"] == current_player)
        ) & (~pending_d["confA"])
        |
        (
            (pending_d["B1"] == current_player) | (pending_d["B2"] == current_player)
        ) & (~pending_d["confB"])
    ]
    # Rundlauf
    rp = pending_r[
        pending_r["Teilnehmer"].str.contains(current_player, na=False)
    ].copy()
    rp["confirmed"] = rp["confirmed_by"].fillna("").apply(lambda s: current_player in s.split(";"))
    rp = rp[~rp["confirmed"]]
    total_pending = len(sp) + len(dp) + len(rp)

    bcols = st.columns(4)
    if bcols[0].button("➕ Einzel", use_container_width=True):
        _open_modal("show_single_modal"); st.rerun()
    if bcols[1].button("➕ Doppel", use_container_width=True):
        _open_modal("show_double_modal"); st.rerun()
    if bcols[2].button("➕ Rundlauf", use_container_width=True):
        _open_modal("show_round_modal"); st.rerun()
    confirm_label = f"✅ Offene bestätigen ({total_pending})" if total_pending > 0 else "✅ Offene bestätigen"
    if bcols[3].button(confirm_label, use_container_width=True):
        _open_modal("show_confirm_modal"); st.rerun()

    # --- Modale Eingabe-Dialoge (Einzel/Doppel/Rundlauf) -----------------
    if st.session_state.show_single_modal:
        with ui_container("Einzelmatch eintragen"):
            st.write(f"Eingeloggt als **{current_player}**")
            pa = st.number_input("Deine Punkte", 0, 21, 11, key="m_pa")
            gegner = st.selectbox("Gegner wählen",
                                  players[players.Name != current_player]["Name"],
                                  index=None, placeholder="Spieler wählen", key="m_opponent")
            pb = st.number_input("Punkte Gegner", 0, 21, 8, key="m_pb")

            col_ok, col_cancel = st.columns(2)
            if col_ok.button("Speichern", key="single_save",
                             disabled=(gegner is None)):
                if gegner == current_player:
                    st.error("Gegner darf nicht identisch sein.")
                else:
                    pending.loc[len(pending)] = [
                        datetime.now(ZoneInfo("Europe/Berlin")),
                        current_player, gegner, pa, pb, True, False
                    ]
                    save_csv(pending, PENDING)
                    st.success("Match gespeichert – wartet auf Bestätigung.")
                    st.rerun()
            if col_cancel.button("Abbrechen", key="single_cancel"):
                _open_modal("")
                st.rerun()

    if st.session_state.show_double_modal:
        with ui_container("Doppelmatch eintragen"):
            st.write("Spieler auswählen")
            a1 = st.selectbox("A1", players.Name, index=None, placeholder="Spieler wählen", key="md_a1")
            a2 = st.selectbox("A2", players[players.Name != a1].Name, index=None, placeholder="Spieler wählen", key="md_a2")
            b1 = st.selectbox("B1", players[~players.Name.isin([a1,a2])].Name, index=None, placeholder="Spieler wählen", key="md_b1")
            b2 = st.selectbox("B2", players[~players.Name.isin([a1,a2,b1])].Name, index=None, placeholder="Spieler wählen", key="md_b2")
            pA = st.number_input("Punkte Team A", 0, 21, 11, key="md_pA")
            pB = st.number_input("Punkte Team B", 0, 21, 8,  key="md_pB")

            disabled = any(x is None for x in (a1,a2,b1,b2))
            c_ok, c_cancel = st.columns(2)
            if c_ok.button("Speichern", key="double_save", disabled=disabled):
                pending_d.loc[len(pending_d)] = [
                    datetime.now(ZoneInfo("Europe/Berlin")), a1, a2, b1, b2, pA, pB, True, False
                ]
                save_csv(pending_d, PENDING_D)
                st.success("Doppel gespeichert – wartet auf Bestätigung.")
                st.rerun()
            if c_cancel.button("Abbrechen", key="double_cancel"):
                _open_modal(""); st.rerun()

    if st.session_state.show_round_modal:
        with ui_container("Rundlauf eintragen"):
            teilnehmer = st.multiselect("Teilnehmer (min. 3)", players.Name, key="mr_teil")
            if len(teilnehmer) >= 3:
                fin_cols = st.columns(2)
                fin1 = fin_cols[0].selectbox("Finalist 1", teilnehmer, index=None, placeholder="Spieler wählen", key="mr_f1")
                fin2_opts = [p for p in teilnehmer if p != fin1]
                fin2 = fin_cols[1].selectbox("Finalist 2", fin2_opts, index=None, placeholder="Spieler wählen", key="mr_f2")
                sieger_opts = [p for p in (fin1, fin2) if p]
                sieger = st.selectbox("Sieger", sieger_opts, index=None, placeholder="Sieger wählen", key="mr_win")
            else:
                fin1 = fin2 = sieger = None

            r_ok, r_cancel = st.columns(2)
            if r_ok.button("Speichern", key="round_save", disabled=(sieger is None)):
                pending_r.loc[len(pending_r)] = {
                    "Datum": datetime.now(ZoneInfo("Europe/Berlin")),
                    "Teilnehmer": ";".join(teilnehmer),
                    "Finalist1": fin1,
                    "Finalist2": fin2,
                    "Sieger": sieger,
                    "creator": current_player,
                    "confirmed_by": current_player,
                }
                save_csv(pending_r, PENDING_R)
                st.success("Rundlauf gespeichert – wartet auf Bestätigung.")
                st.rerun()
            if r_cancel.button("Abbrechen", key="round_cancel"):
                _open_modal("")
                st.rerun()

    # --- Modal: Offene Matches bestätigen ----------------------------------
    if st.session_state.show_confirm_modal:
        with ui_container("Offene Matches bestätigen"):
            st.write("### Einzel")
            for idx, row in pending.iterrows():
                needs_me = (
                    (row["A"] == current_player and not row["confA"])
                    or (row["B"] == current_player and not row["confB"])
                )
                if not needs_me:
                    continue
                col1, col_ok, col_rej = st.columns([3, 1, 1])
                col1.write(f"{row['A']} {int(row['PunkteA'])} : {int(row['PunkteB'])} {row['B']}")
                if col_ok.button("✅", key=f"conf_single_{idx}"):
                    if row["A"] == current_player:
                        pending.at[idx, "confA"] = True
                    else:
                        pending.at[idx, "confB"] = True
                    if pending.at[idx, "confA"] and pending.at[idx, "confB"]:
                        matches.loc[len(matches)] = pending.loc[idx, pending.columns[:-2]]
                        pending.drop(idx, inplace=True)
                        save_csv(matches, MATCHES)
                        _rebuild_all()
                    save_csv(pending, PENDING)
                    st.success("Bestätigt.")
                    st.rerun()
                if col_rej.button("❌", key=f"rej_single_{idx}"):
                    pending.drop(idx, inplace=True)
                    save_csv(pending, PENDING)
                    st.warning("Match abgelehnt und entfernt.")
                    st.rerun()

            st.write("### Doppel")
            for idx, row in pending_d.iterrows():
                in_team_a = current_player in (row["A1"], row["A2"])
                in_team_b = current_player in (row["B1"], row["B2"])
                if not (in_team_a or in_team_b):
                    continue
                needs_me = (
                    (in_team_a and not row["confA"])
                    or (in_team_b and not row["confB"])
                )
                if not needs_me:
                    continue
                col1, col_ok, col_rej = st.columns([3, 1, 1])
                teams = (
                    f"{row['A1']} / {row['A2']}  {int(row['PunkteA'])} : "
                    f"{int(row['PunkteB'])}  {row['B1']} / {row['B2']}"
                )
                col1.write(teams)
                if col_ok.button("✅", key=f"conf_double_{idx}"):
                    if in_team_a:
                        pending_d.at[idx, "confA"] = True
                    else:
                        pending_d.at[idx, "confB"] = True
                    if pending_d.at[idx, "confA"] and pending_d.at[idx, "confB"]:
                        doubles.loc[len(doubles)] = pending_d.loc[idx, pending_d.columns[:-2]]
                        pending_d.drop(idx, inplace=True)
                        save_csv(doubles, DOUBLES)
                        _rebuild_all()
                    save_csv(pending_d, PENDING_D)
                    st.success("Bestätigt.")
                    st.rerun()
                if col_rej.button("❌", key=f"rej_double_{idx}"):
                    pending_d.drop(idx, inplace=True)
                    save_csv(pending_d, PENDING_D)
                    st.warning("Doppel abgelehnt und entfernt.")
                    st.rerun()

            st.write("### Rundlauf")
            for idx, row in pending_r.iterrows():
                teilnehmer = row["Teilnehmer"].split(";")
                if current_player not in teilnehmer:
                    continue
                confirmed = set(row["confirmed_by"].split(";")) if row["confirmed_by"] else set()
                if current_player in confirmed:
                    continue
                col1, col_ok, col_rej = st.columns([3, 1, 1])
                col1.write(f"{', '.join(teilnehmer)}  –  Sieger: {row['Sieger']}")
                if col_ok.button("✅", key=f"conf_round_{idx}"):
                    confirmed.add(current_player)
                    pending_r.at[idx, "confirmed_by"] = ";".join(sorted(confirmed))
                    if len(confirmed) >= 3:
                        rounds.loc[len(rounds)] = pending_r.loc[idx, pending_r.columns[:-1]]
                        pending_r.drop(idx, inplace=True)
                        save_csv(rounds, ROUNDS)
                        _rebuild_all()
                    save_csv(pending_r, PENDING_R)
                    st.success("Bestätigt.")
                    st.rerun()
                if col_rej.button("❌", key=f"rej_round_{idx}"):
                    pending_r.drop(idx, inplace=True)
                    save_csv(pending_r, PENDING_R)
                    st.warning("Rundlauf abgelehnt und entfernt.")
                    st.rerun()

            if st.button("❌ Schließen"):
                _open_modal("")
                st.rerun()

    st.stop()

# endregion

# region Regel Ansicht
if st.session_state.view_mode == "regeln":
    rules_html = """
    <style>
    .rulebox {font-size:18px; line-height:1.45;}
    .rulebox h2 {font-size:24px; margin:1.2em 0 .5em;}
    .rulebox h3 {font-size:20px; margin:1.0em 0 .3em;}
    .rulebox ul {margin:0 0 1em 1.3em; list-style:disc;}
    </style>

    <div class="rulebox">

    <h2>Einzelmatch</h2>

    <h3>1.&nbsp;Spielziel:</h3>
    <p>Wer zuerst 11&nbsp;Punkte (mit mindestens&nbsp;2 Punkten Vorsprung) erreicht, gewinnt das Match.</p>

    <h3>2.&nbsp;Aufschlag&nbsp;&amp;&nbsp;Rückschlag:</h3>
    <p>
    Der Aufschlag beginnt offen (sichtbar) und wird vom eigenen Spielfeld auf das gegnerische Feld gespielt.<br>
    Der Ball muss dabei einmal auf der eigenen Seite und dann einmal auf der gegnerischen Seite aufkommen.<br>
    Nach dem Aufschlag erfolgt der Rückschlag: Der Ball wird direkt auf die gegnerische Seite geschlagen
    (nicht mehr auf der eigenen aufkommen lassen).
    </p>

    <h3>3.&nbsp;Rallye:</h3>
    <p>
    Nach dem Aufschlag wechseln sich die Spieler ab.<br>
    Der Ball darf maximal einmal aufspringen, muss über oder um das Netz geschlagen werden.<br>
    Berührt der Ball das Netz beim Rückschlag, aber landet korrekt, wird weitergespielt.<br>
    Beim Aufschlag hingegen führt Netzberührung bei korrektem Verlauf zu einem „Let“ (Wiederholung des Aufschlags).
    </p>

    <h3>4.&nbsp;Punktevergabe:</h3>
    <ul>
      <li>Aufschlagfehler (z.&nbsp;B. Ball landet nicht auf gegnerischer Seite)</li>
      <li>Ball verfehlt</li>
      <li>Ball springt zweimal auf der eigenen Seite</li>
      <li>Rückschlag landet außerhalb oder im Netz</li>
      <li>Ball wird vor dem Aufspringen, aber über der Tischfläche getroffen</li>
      <li>Netz oder Tisch wird mit der Hand oder dem Körper berührt</li>
    </ul>

    <h3>5.&nbsp;Aufschlagwechsel:</h3>
    <p>
    Alle&nbsp;2 Punkte wird der Aufschlag gewechselt.<br>
    Bei 10&nbsp;:&nbsp;10 wird nach jedem Punkt der Aufschlag gewechselt
    (bis einer 2&nbsp;Punkte Vorsprung hat).
    </p>

    <h3>6.&nbsp;Seitenwechsel:</h3>
    <p>
    Nach jedem Satz werden die Seiten gewechselt.<br>
    Im Entscheidungssatz (z.&nbsp;B. 5.&nbsp;Satz bei 3&nbsp;:&nbsp;2) zusätzlich bei 5 Punkten.
    </p>

    </div>

    <div class="rulebox">

    <h2>Doppelmatch</h2>

    <h3>1.&nbsp;Spielziel:</h3>
    <p>Wie beim Einzel gilt: 11&nbsp;Punkte mit mindestens 2 Punkten Vorsprung.</p>

    <h3>2.&nbsp;Aufschlag&nbsp;&amp;&nbsp;Rückschlag:</h3>
    <p>
    Aufschlag erfolgt immer diagonal von der rechten Hälfte zur rechten Hälfte des Gegners.<br>
    Reihenfolge: A1 schlägt auf B1 auf, dann B1 auf A2, dann A2 auf B2, dann B2 auf A1 usw.
    </p>

    <h3>3.&nbsp;Schlagreihenfolge:</h3>
    <p>
    Innerhalb eines Ballwechsels muss sich jedes Team beim Schlagen abwechseln.<br>
    Es darf also nicht zweimal hintereinander vom selben Spieler eines Teams gespielt werden.
    </p>

    <h3>4.&nbsp;Punktevergabe:</h3>
    <p>Fehler führen wie im Einzelspiel zu einem Punkt für das gegnerische Team:</p>
    <ul>
      <li>Aufschlagfehler (z.&nbsp;B. falsches Feld, Netz)</li>
      <li>Ball ins Aus oder ins Netz</li>
      <li>Fehlerhafte Reihenfolge beim Schlagen</li>
      <li>Berührung von Netz oder Tisch mit dem Körper</li>
    </ul>

    <h3>5.&nbsp;Aufschlagwechsel:</h3>
    <p>
    Der Aufschlag wechselt alle 2 Punkte – dabei ändert sich auch die Reihenfolge der Spieler.<br>
    Nach jedem Satz rotiert die Reihenfolge, sodass jeder mal mit jedem spielt.
    </p>

    <h3>6.&nbsp;Seitenwechsel:</h3>
    <p>Wie im Einzel: Nach jedem Satz und bei 5 Punkten im letzten Satz.</p>

    </div>

    <div class="rulebox">

    <h2>Rundlauf</h2>

    <h3>1.&nbsp;Spielprinzip:</h3>
    <p>
    Alle Spieler laufen im Kreis um den Tisch. Jeder darf nur einen Schlag ausführen und muss danach sofort weiterlaufen.<br>
    Wer einen Fehler macht, verliert ein Leben.
    </p>

    <h3>2.&nbsp;Leben &amp; Ausscheiden:</h3>
    <p>
    Jeder Spieler startet mit <strong>3 Leben</strong>. Wer keine Leben mehr hat, scheidet aus.<br>
    Der erste Spieler, der ausscheidet, bekommt automatisch einen <strong>„Schwimmer“</strong> (ein zusätzliches Leben) und darf wieder mitspielen.
    </p>

    <h3>3.&nbsp;Finalrunde:</h3>
    <p>
    Wenn nur noch zwei Spieler übrig sind, beginnt das <strong>Finale</strong>.<br>
    Es wird auf Punkte gespielt: <strong>Bis 3 Punkte mit mindestens 2 Punkten Abstand</strong>.
    </p>

    <h3>4.&nbsp;Aufschlag im Finale:</h3>
    <p>
    Im Finale wechselt der Aufschlag nach jedem Punkt. Es wird ganz normal mit einem regulären Aufschlag begonnen.
    </p>

    <h3>5.&nbsp;Fehlerquellen:</h3>
    <ul>
      <li>Ball im Netz oder im Aus</li>
      <li>Ball springt nicht auf gegnerischer Seite</li>
      <li>Doppelberührung oder Schlag vor dem Aufspringen</li>
      <li>Zu spät am Tisch (wenn Ball schon 2× aufspringt oder Mitspieler warten muss)</li>
    </ul>

    </div>
    """
    st.markdown(rules_html, unsafe_allow_html=True)
    st.stop()
# endregion
