# region Imports
import streamlit as st
 # Admin user(s) for manual full rebuild
ADMINS = ["Chris"]
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
# Turniermodus
TOURNAMENTS = Path("tournaments.csv")
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

        # live / dev umschalten:
        # return gc_local.open("tt_elo")    # Produktion
        return gc_local.open("ttelodev")    # Dev-Umgebung

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

# Turniere laden
tournaments = load_or_create(TOURNAMENTS, ["ID","Name","Creator","Time","Note","Limit","Teilnehmer"])
if "Teilnehmer" not in tournaments.columns:
    tournaments["Teilnehmer"] = ""
    save_csv(tournaments, TOURNAMENTS)

# Ensure confA/confB are boolean for logical operations
pending["confA"]   = pending["confA"].astype(bool)
pending["confB"]   = pending["confB"].astype(bool)
pending_d["confA"] = pending_d["confA"].astype(bool)
pending_d["confB"] = pending_d["confB"].astype(bool)

# Always convert Datum to pandas datetime with Berlin timezone
for df in (matches, pending, pending_d, doubles, pending_r, rounds):
    df["Datum"] = pd.to_datetime(df["Datum"], utc=True, errors="coerce")\
                       .dt.tz_convert("Europe/Berlin")
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
            # Manuelle Eingabe des Spielernamens statt Auswahl
            login_name = st.text_input("Spielername")
            login_pin = st.text_input("PIN", type="password")
            if st.button("Einloggen"):
                if login_name not in players["Name"].values:
                    st.error("Spielername nicht gefunden.")
                else:
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
        
        if st.button("🏆 Turniermodus", use_container_width=True):
            st.session_state.view_mode = "turniermodus"
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

        # Admin: vollständigen Rebuild auslösen
        if current_player in ADMINS:
            if st.button("🔄 Admin: Alle ELO neu berechnen", use_container_width=True):
                _rebuild_all()
                st.success("Alle Elo-Werte neu berechnet.")
                st.rerun()


if not st.session_state.logged_in:
    st.stop()
#
# endregion


current_player = st.session_state.current_player

# region Home Ansicht
if st.session_state.view_mode == "home":
    # Prepare tabs: Willkommen, Matches, Statistiken
    tab1, tab2, tab3 = st.tabs(["Willkommen", "Matches", "Statistiken"])

    # Daten für aktuellen Spieler und offene Bestätigungen
    user = players.loc[players.Name == current_player].iloc[0]
    sp = pending[
        ((pending["A"] == current_player) & (~pending["confA"]))
        | ((pending["B"] == current_player) & (~pending["confB"]))
    ]
    dp = pending_d[
        (
            ((pending_d["A1"] == current_player) | (pending_d["A2"] == current_player))
            & (~pending_d["confA"])
        )
        | (
            ((pending_d["B1"] == current_player) | (pending_d["B2"] == current_player))
            & (~pending_d["confB"])
        )
    ]
    rp = pending_r[pending_r["Teilnehmer"].str.contains(current_player, na=False)].copy()
    rp["confirmed"] = rp["confirmed_by"].fillna("").apply(lambda s: current_player in s.split(";"))
    rp = rp[~rp["confirmed"]]
    total_pending = len(sp) + len(dp) + len(rp)

    # Tab 1: Willkommen
    with tab1:
        st.markdown(
            f'<h3 style="text-align:center;">Willkommen, <strong>{current_player}</strong>!</h3>',
            unsafe_allow_html=True
        )
        st.markdown(
            f"**Gesamt-ELO:** {int(user.G_ELO)}  |  Einzel: {int(user.ELO)}  |  Doppel: {int(user.D_ELO)}  |  Rundlauf: {int(user.R_ELO)}"
        )
        # Button: offene Matches bestätigen
        if total_pending > 0:
            if st.button(f"✅ Offene Matches bestätigen ({total_pending})", use_container_width=True):
                _open_modal("show_confirm_modal")
                st.rerun()
        else:
            st.button("✅ Offene Matches bestätigen", disabled=True)
        # Letzte 5 Einzel-Matches anzeigen
        recent = matches[
            (matches["A"] == current_player) | (matches["B"] == current_player)
        ].sort_values("Datum", ascending=False).head(5)
        if not recent.empty:
            st.subheader("Letzte 5 Spiele")
            st.table(recent[["Datum", "A", "PunkteA", "PunkteB", "B"]])

    # Tab 2: Match-Eintrag und Bestätigung (wie bisher)
    with tab2:
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
        # Die bestehenden Modal-Dialoge (_open_modal) bleiben unverändert

    # Tab 3: Leaderboards und Statistiken (wie bisher)
    with tab3:
        st.subheader("ELO-Übersicht")
        # CSS, um die Index-Spalte in statischen Tabellen auszublenden
        st.markdown(
            """
            <style>
            .stTable table tr th:first-child,
            .stTable table tr td:first-child {
                display: none;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        # Unter-Tabs für verschiedene Sortierungen
        sub_tabs = st.tabs(["Gesamt", "Einzel", "Doppel", "Rundlauf"])
        # CSS, um die Tabs gleichmäßig über die Breite zu verteilen
        st.markdown(
            """
            <style>
            [role="tablist"] > [role="tab"] {
                flex: 1 !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        for idx, metric in enumerate(["G_ELO", "ELO", "D_ELO", "R_ELO"]):
            with sub_tabs[idx]:
                # Tabelle vorbereiten
                df_tab = players[["Name", "G_ELO", "ELO", "D_ELO", "R_ELO"]].copy()
                df_tab = df_tab.rename(columns={
                    "G_ELO": "Gesamt",
                    "ELO": "Einzel",
                    "D_ELO": "Doppel",
                    "R_ELO": "Rundlauf"
                })
                # Ganzzahlige ELO-Werte
                for col in ["Gesamt", "Einzel", "Doppel", "Rundlauf"]:
                    df_tab[col] = df_tab[col].astype(int)
                # Nach gewähltem Metric absteigend sortieren
                col_name = ["Gesamt", "Einzel", "Doppel", "Rundlauf"][idx]
                df_tab = df_tab.sort_values(col_name, ascending=False)
                # Statische Tabelle mit Hervorhebung des aktuellen Spielers, Index versteckt
                df_disp = df_tab.reset_index(drop=True)
                # Highlight aktueller Spieler
                def highlight_current(row):
                    return [
                        "background-color: #ADD8E6; color: black"
                        if row["Name"] == current_player else ""
                        for _ in row
                    ]
                styler_disp = df_disp.style.apply(highlight_current, axis=1)
                # Index-Spalte in der gerenderten Tabelle ausblenden
                styler_disp = styler_disp.set_table_styles([
                    {"selector": "th.row_heading, td.row_heading", "props": [("display", "none")]},
                    {"selector": "th.blank.level0", "props": [("display", "none")]}
                ])
                st.table(styler_disp)

        # Persönliche Statistiken in kompakter Tabelle (Kategorie, Spiele, Siege, Winrate)
        st.markdown("---")
        st.subheader("Meine Statistiken")
        # DataFrame mit persönlichen Werten
        stats = pd.DataFrame({
            "Kategorie": ["Einzel", "Doppel", "Rundlauf", "Gesamt"],
            "Spiele": [
                int(user.Spiele),
                int(user.D_Spiele),
                int(user.R_Spiele),
                int(user.Spiele + user.D_Spiele + user.R_Spiele)
            ],
            "Siege": [
                int(user.Siege),
                int(user.D_Siege),
                int(user.R_Siege),
                int(user.Siege + user.D_Siege + user.R_Siege)
            ]
        })
        # Winrate in Prozent berechnen (ganzzahlig) mit Prozentzeichen
        stats["Winrate"] = stats.apply(
            lambda row: f"{int(round((row['Siege'] / row['Spiele']) * 100))}%" if row["Spiele"] > 0 else "0%",
            axis=1
        )
        # Statische Tabelle ohne Index
        st.table(stats)


    st.stop()
# endregion

 # region Regel Ansicht
if st.session_state.view_mode == "regeln":
    rules_html = """
    <style>
    .rulebox {
      font-size: 18px;
      line-height: 1.45;
      padding: 1rem;
      border-radius: 8px;
      background-color: rgba(255,255,255,0.9);
      color: #000;
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
      border: 1px solid rgba(0, 0, 0, 0.1);
    }
    .rulebox h2 {
      font-size: 24px;
      margin: 1.2em 0 0.5em;
    }
    .rulebox h3 {
      font-size: 20px;
      margin: 1em 0 0.3em;
    }
    .rulebox ul {
      margin: 0 0 1em 1.3em;
      list-style: disc;
    }
    @media (prefers-color-scheme: dark) {
      .rulebox {
        background-color: rgba(33,33,33,0.85);
        color: #fff;
        box-shadow: 0 2px 6px rgba(255, 255, 255, 0.15);
        border: 1px solid #ffffff !important;
      }
    }
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

# region Turniermodus Ansicht
# region Turniermodus Ansicht
if st.session_state.view_mode == "turniermodus":
    st.markdown(
        '<h1 style="text-align:center; margin-bottom:0.25rem;">🏆 Turniermodus</h1>',
        unsafe_allow_html=True
    )
    # Expander zum Erstellen eines Turniers
    with st.expander("Turnier erstellen", expanded=True):
        turnier_name = st.text_input("Turniername")
        # Datum und Uhrzeit separat erfassen
        turnier_date = st.date_input(
            "Datum", value=datetime.now(ZoneInfo("Europe/Berlin")).date()
        )
        turnier_time_input = st.time_input(
            "Uhrzeit", value=datetime.now(ZoneInfo("Europe/Berlin")).timetz().replace(second=0, microsecond=0)
        )
        turnier_time = datetime.combine(turnier_date, turnier_time_input).astimezone(ZoneInfo("Europe/Berlin"))
        turnier_note = st.text_area("Notiz (optional)")
        turnier_limit = st.number_input(
            "Maximale Teilnehmer (0 = unbegrenzt)",
            min_value=0, value=0, step=1
        )
        if st.button("Turnier erstellen"):
            import uuid
            tid = uuid.uuid4().hex[:8]
            # Ensure Started column exists
            if "Started" not in tournaments.columns:
                tournaments["Started"] = False
            tournaments.loc[len(tournaments)] = [
                tid,
                turnier_name,
                current_player,
                turnier_time.isoformat(),
                turnier_note,
                int(turnier_limit),
                current_player,
                False
            ]
            save_csv(tournaments, TOURNAMENTS)
            st.success("Turnier erstellt!")
            st.rerun()
    # Expander zum Beitreten bestehender Turniere
    with st.expander("Turnier beitreten", expanded=False):
        if tournaments.empty:
            st.info("Noch keine Turniere verfügbar.")
        else:
            for idx, row in tournaments.iterrows():
                parts = row["Teilnehmer"].split(";") if row["Teilnehmer"] else []
                cols = st.columns([3,1])
                # Turnier-Info
                cols[0].markdown(
                    f"**{row['Name']}** von {row['Creator']}<br>"
                    f"{row['Time']}<br>"
                    f"Teilnehmer: {len(parts)}"
                    , unsafe_allow_html=True
                )
                # Join-Button, Voll-Status oder schon beigetreten
                if current_player not in parts:
                    if row["Limit"] == 0 or len(parts) < row["Limit"]:
                        if cols[1].button("Beitreten", key=f"join_{idx}"):
                            parts.append(current_player)
                            tournaments.at[idx, "Teilnehmer"] = ";".join(parts)
                            save_csv(tournaments, TOURNAMENTS)
                            st.success("Beigetreten!")
                            st.rerun()
                    else:
                        cols[1].write("🔒 Voll")
                else:
                    cols[1].write("✅ Beigetreten")
                # Ersteller kann ab 4 Teilnehmern starten
                if row["Creator"] == current_player and len(parts) >= 4 and not row.get("Started", False):
                    if cols[1].button("Turnier starten", key=f"start_{idx}"):
                        tournaments.at[idx, "Started"] = True
                        save_csv(tournaments, TOURNAMENTS)
                        st.success("Turnier gestartet!")
                        st.rerun()
    st.stop()
# endregion

