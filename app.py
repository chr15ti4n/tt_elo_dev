# region Imports
import streamlit as st
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

# region Admins
ADMINS = {"Chris"}
# endregion

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
        ws.clear()
        set_with_dataframe(ws, df.reset_index(drop=True))
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
def rebuild_players(players_df: pd.DataFrame, matches_df: pd.DataFrame, k: int = 32) -> pd.DataFrame:
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
        margin = pa - pb
        score_a = max(margin / 11, 0)            # 11:0 -> 1   11:10 -> 0.09
        score_b = max(-margin / 11, 0)           # Gegenwert für Verlierer

        new_r_a = calc_elo(r_a, r_b, score_a, k)
        new_r_b = calc_elo(r_b, r_a, score_b, k)

        players_df.loc[players_df["Name"] == a, ["ELO", "Siege", "Niederlagen", "Spiele"]] = [
            new_r_a,
            players_df.loc[players_df["Name"] == a, "Siege"].iat[0] + score_a,
            players_df.loc[players_df["Name"] == a, "Niederlagen"].iat[0] + score_b,
            players_df.loc[players_df["Name"] == a, "Spiele"].iat[0] + 1,
        ]
        players_df.loc[players_df["Name"] == b, ["ELO", "Siege", "Niederlagen", "Spiele"]] = [
            new_r_b,
            players_df.loc[players_df["Name"] == b, "Siege"].iat[0] + score_b,
            players_df.loc[players_df["Name"] == b, "Niederlagen"].iat[0] + score_a,
            players_df.loc[players_df["Name"] == b, "Spiele"].iat[0] + 1,
        ]
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
for df in (matches, pending, pending_d, doubles, pending_r, rounds):
    if not df.empty:
        df["Datum"] = (
            pd.to_datetime(df["Datum"], utc=True, errors="coerce")
              .dt.tz_convert("Europe/Berlin")
        )
# endregion

# region Doppel ELO Rebuild
# ---------- Doppel-Stats & ELO komplett neu berechnen ----------
def rebuild_players_d(players_df, doubles_df, k=24):
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
        margin = pa - pb
        score_a = max(margin / 11, 0)
        nr1,nr2 = calc_doppel_elo(ra1,ra2,b_avg,score_a,k)
        nr3,nr4 = calc_doppel_elo(rb1,rb2,a_avg,1-score_a,k)
        updates = [(a1,nr1,score_a),(a2,nr2,score_a),(b1,nr3,1-score_a),(b2,nr4,1-score_a)]
        for p,new,s in updates:
            players_df.loc[players_df.Name==p, ["D_ELO","D_Siege","D_Niederlagen","D_Spiele"]] = [
                new,
                players_df.loc[players_df.Name==p,"D_Siege"].iat[0] + (1 if s==1 else 0),
                players_df.loc[players_df.Name==p,"D_Niederlagen"].iat[0] + (1 if s==0 else 0),
                players_df.loc[players_df.Name==p,"D_Spiele"].iat[0] + 1,
            ]
    return players_df
# endregion

# region Rundlauf ELO
# ---------- Rundlauf-ELO ----------
def calc_round_elo(r, avg, s, k=24):
    """
    r      ... Rating des Spielers
    avg    ... Durchschnittsrating aller Teilnehmer
    s      ... Ergebnis (1 Sieger, 0.5 Zweiter, 0 Verlierer)
    """
    exp = 1 / (1 + 10 ** ((avg - r) / 400))
    return round(r + k * (s - exp))

# ---------- Rundlauf-Stats & ELO komplett neu berechnen ----------
def rebuild_players_r(players_df, rounds_df, k=24):
    players_df = players_df.copy()
    players_df[["R_ELO","R_Siege","R_Zweite","R_Niederlagen","R_Spiele"]] = 0
    players_df["R_ELO"] = 1200
    if rounds_df.empty:
        return players_df
    for _, row in rounds_df.sort_values("Datum").iterrows():
        teilnehmer = row["Teilnehmer"].split(";")
        fin1, fin2, winner = row["Finalist1"], row["Finalist2"], row["Sieger"]
        avg = players_df.loc[players_df.Name.isin(teilnehmer), "R_ELO"].mean()
        for p in teilnehmer:
            old = players_df.loc[players_df.Name==p,"R_ELO"].iat[0]
            if p == winner:
                s = 1
                players_df.loc[players_df.Name==p,"R_Siege"] += 1
            elif p in (fin1, fin2):
                s = 0.5   # Finalist (2. Platz) erhält halben Sieg‑Wert
                players_df.loc[players_df.Name==p,"R_Zweite"] += 1
            else:
                s = 0
                players_df.loc[players_df.Name==p,"R_Niederlagen"] += 1
            new = calc_round_elo(old, avg, s, k)
            players_df.loc[players_df.Name==p,"R_ELO"] = new
            players_df.loc[players_df.Name==p,"R_Spiele"] += 1
    return players_df
# endregion


# region Auth & Sidebar UI
# ---------- Login / Registrierung ----------
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
    st.session_state.view_mode = "spiel"

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
                    "D_ELO": 1200,
                    "D_Siege": 0,
                    "D_Niederlagen": 0,
                    "D_Spiele": 0,
                }
                players = pd.concat([players, pd.DataFrame([new_player])], ignore_index=True)
                save_csv(players, PLAYERS)
                st.success(f"{reg_name} angelegt. Jetzt einloggen.")
                st.rerun()
# Eingeloggt: Sidebar zeigt Menü und Logout
else:
    with st.sidebar:
        current_player = st.session_state.current_player  # lokal verfügbar
        st.markdown(f"**Eingeloggt als:** {current_player}")

        if st.button("🏓 Einzelmatch", use_container_width=True):
            st.session_state.view_mode = "spiel"
            st.rerun()

        if st.button("🤝 Doppel", use_container_width=True):
            st.session_state.view_mode = "doppel"
            st.rerun()

        if st.button("🔄 Rundlauf", use_container_width=True):
            st.session_state.view_mode = "round"
            st.rerun()

        if st.button("📜 Regeln", use_container_width=True):
            st.session_state.view_mode = "regeln"
            st.rerun()

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.current_player = None
            st.query_params.clear()  # clear
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

        # Admin‑Tools (nur für Admins sichtbar)
        if current_player in ADMINS:
            with st.expander("⚙️ Admin"):
                st.write("Eintrag auswählen und entfernen – ELO wird automatisch neu berechnet.")
                mode = st.selectbox("Typ", ["Einzel", "Doppel", "Rundlauf"])
                if mode == "Einzel" and not matches.empty:
                    idx = st.number_input("Index (Einzel)", 0, len(matches)-1, step=1)
                elif mode == "Doppel" and not doubles.empty:
                    idx = st.number_input("Index (Doppel)", 0, len(doubles)-1, step=1)
                elif mode == "Rundlauf" and not rounds.empty:
                    idx = st.number_input("Index (Rundlauf)", 0, len(rounds)-1, step=1)
                else:
                    st.info("Keine Einträge vorhanden.")
                    idx = None

                if st.button("Löschen") and idx is not None:
                    if mode == "Einzel":
                        matches.drop(idx, inplace=True)
                        matches.reset_index(drop=True, inplace=True)
                        players_updated = rebuild_players(players, matches)
                        players.update(players_updated)
                        save_csv(matches, MATCHES)
                    elif mode == "Doppel":
                        doubles.drop(idx, inplace=True)
                        doubles.reset_index(drop=True, inplace=True)
                        players_updated = rebuild_players_d(players, doubles)
                        players.update(players_updated)
                        save_csv(doubles, DOUBLES)
                    elif mode == "Rundlauf":
                        rounds.drop(idx, inplace=True)
                        rounds.reset_index(drop=True, inplace=True)
                        players_updated = rebuild_players_r(players, rounds)
                        players.update(players_updated)
                        save_csv(rounds, ROUNDS)

                    save_csv(players, PLAYERS)
                    st.success(f"{mode}-Eintrag Nr. {idx} gelöscht und Elo neu berechnet.")
                    st.rerun()

# Login erforderlich, um fortzufahren
if not st.session_state.logged_in:
    st.stop()
#
# endregion

current_player = st.session_state.current_player

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

# region Doppel Ansicht
if st.session_state.view_mode == "doppel":
    st.title("Doppelmatch eintragen")
    if len(players) < 4:
        st.info("Mindestens vier Spieler registrieren.")
    else:
        c = st.columns(4)
        a1 = c[0].selectbox("A1", players.Name, index=None, placeholder="Spieler wählen", key="d_a1")
        a2 = c[1].selectbox("A2", players[players.Name != a1].Name, index=None, placeholder="Spieler wählen", key="d_a2")
        b1 = c[2].selectbox("B1", players[~players.Name.isin([a1,a2])].Name, index=None, placeholder="Spieler wählen", key="d_b1")
        b2 = c[3].selectbox("B2", players[~players.Name.isin([a1,a2,b1])].Name, index=None, placeholder="Spieler wählen", key="d_b2")
        pA = st.number_input("Punkte A", 0, 21, 11, key="d_pA")
        pB = st.number_input("Punkte B", 0, 21, 8, key="d_pB")
        save_disabled = any(x is None for x in (a1, a2, b1, b2))
        if st.button("Doppel speichern", disabled=save_disabled):
            pending_d.loc[len(pending_d)] = [
                datetime.now(ZoneInfo("Europe/Berlin")), a1, a2, b1, b2, pA, pB, True, False
            ]
            save_csv(pending_d, PENDING_D)
            st.success("Doppel gespeichert – wartet auf Bestätigung.")

    # Bestätigungs‑Expander
    with st.expander("Offene Doppel"):
        to_conf = pending_d[((pending_d.A1==current_player)|(pending_d.A2==current_player)) & (~pending_d.confA) |
                            ((pending_d.B1==current_player)|(pending_d.B2==current_player)) & (~pending_d.confB)]
        if to_conf.empty:
            st.info("Keine offenen Doppel")
        else:
            for idx,row in to_conf.iterrows():
                st.write(f"{row.A1}&{row.A2}  {row.PunkteA}:{row.PunkteB}  {row.B1}&{row.B2}")
                col1,col2 = st.columns(2)
                if col1.button("Bestätigen ✅", key=f"d_ok_{idx}"):
                    pending_d.at[idx, "confA" if current_player in (row.A1,row.A2) else "confB"] = True
                    if pending_d.at[idx,"confA"] and pending_d.at[idx,"confB"]:
                        doubles = pd.concat(
                            [doubles, pending_d.loc[[idx], pending_d.columns[:-2]]],
                            ignore_index=True
                        )
                        pending_d.drop(idx, inplace=True)
                        players = rebuild_players_d(players, doubles)
                        save_csv(doubles, DOUBLES)
                    save_csv(pending_d, PENDING_D)
                    save_csv(players, PLAYERS)
                    st.rerun()
                if col2.button("Ablehnen ❌", key=f"d_no_{idx}"):
                    pending_d.drop(idx, inplace=True)
                    save_csv(pending_d, PENDING_D)
                    st.rerun()

    st.subheader("Doppel-Leaderboard")
    d_leader = (
        players[players["D_Spiele"] > 0][["Name","D_ELO","D_Siege","D_Niederlagen","D_Spiele"]]
        .rename(columns={
            "D_ELO": "ELO",
            "D_Siege": "Siege",
            "D_Niederlagen": "Niederlagen",
            "D_Spiele": "Spiele",
        })
        .sort_values("ELO", ascending=False)
        .reset_index(drop=True)
    )
    if d_leader.empty:
        st.info("Noch keine bestätigten Doppelmatches.")
    else:
        st.dataframe(d_leader)

    # ---------- Letzte 5 Doppelmatches ----------
    st.subheader("Letzte 5 Doppelmatches")
    if doubles.empty:
        st.info("Noch keine Doppelmatches eingetragen.")
    else:
        recent_d = (
            doubles.sort_values("Datum", ascending=False)
                   .head(5)
                   .reset_index(drop=True)
        )
        recent_d["Datum"] = recent_d["Datum"].dt.strftime("%d.%m.%Y")
        recent_d_display = recent_d[["Datum", "A1", "A2", "B1", "B2", "PunkteA", "PunkteB"]]
        st.dataframe(recent_d_display)
    st.stop()
# endregion

# region Einzel Ansicht
# Zeige den folgenden Block nur im Einzel‑Modus
if st.session_state.view_mode == "spiel":
    # ---------- Match erfassen ----------
    st.title("AK-Tischtennis")
    st.subheader("Match eintragen")

    if len(players) < 2:
        st.info("Mindestens zwei Spieler registrieren, um ein Match anzulegen.")
    else:
        st.markdown(f"**Eingeloggt als:** {current_player}")
        pa = st.number_input("Punkte (dein Ergebnis)", 0, 21, 11)
        b = st.selectbox("Gegner wählen", players[players["Name"] != current_player]["Name"],
                         index=None, placeholder="Spieler wählen")
        pb = st.number_input("Punkte Gegner", 0, 21, 8)
        save_disabled = (b is None)
        if st.button("Match speichern", disabled=save_disabled):
            if current_player == b:
                st.error("Spieler dürfen nicht identisch sein.")
            else:
                ts_now = datetime.now(ZoneInfo("Europe/Berlin"))
                pending.loc[len(pending)] = [
                    ts_now, current_player, b, pa, pb,
                    True,  # confA (du)
                    False  # confB
                ]
                save_csv(pending, PENDING)
                st.success("Match gespeichert! Es wartet jetzt auf Bestätigung des Gegners.")


    # ---------- Offene Matches bestätigen ----------
    with st.expander("Offene Matches bestätigen"):
        to_confirm = pending[
            ((pending["A"] == current_player) & (pending["confA"] == False)) |
            ((pending["B"] == current_player) & (pending["confB"] == False))
        ]
        if to_confirm.empty:
            st.info("Keine offenen Matches für dich.")
        else:
            for idx, row in to_confirm.iterrows():
                match_text = (f"{row['A']} {row['PunkteA']} : {row['PunkteB']} {row['B']}  "
                              f"({row['Datum'].strftime('%d.%m.%Y %H:%M')})")
                st.write(match_text)

                col_ok, col_rej = st.columns(2)
                with col_ok:
                    if st.button("Bestätigen ✅", key=f"conf_{idx}"):
                        if row["A"] == current_player:
                            pending.at[idx, "confA"] = True
                        else:
                            pending.at[idx, "confB"] = True
                        # Wenn beide bestätigt, Match finalisieren
                        if pending.at[idx, "confA"] and pending.at[idx, "confB"]:
                            matches = pd.concat(
                                [matches,
                                 pending.loc[[idx], ["Datum","A","B","PunkteA","PunkteB"]]],
                                ignore_index=True
                            )
                            pending.drop(idx, inplace=True)
                            players = rebuild_players(players, matches)
                            save_csv(matches, MATCHES)
                        save_csv(pending, PENDING)
                        save_csv(players, PLAYERS)
                        st.rerun()

                with col_rej:
                    if st.button("Ablehnen ❌", key=f"rej_{idx}"):
                        # Einfach aus der Pending-Liste entfernen, keine ELO-Anpassung
                        pending.drop(idx, inplace=True)
                        save_csv(pending, PENDING)
                        st.success("Match abgelehnt und entfernt.")
                        st.rerun()


    # ---------- Leaderboard anzeigen ----------

    # ---------- Leaderboard anzeigen ----------
    st.subheader("Leaderboard")
    ex_cols = ["Pin", "D_ELO", "D_Siege", "D_Niederlagen", "D_Spiele",
               "R_ELO", "R_Siege", "R_Zweite", "R_Niederlagen", "R_Spiele"]
    einzel_tbl = (players[players["Spiele"] > 0]                 # nur wer mind. 1 Einzel spielte
                  .drop(columns=ex_cols, errors="ignore")
                  .sort_values("ELO", ascending=False)
                  .reset_index(drop=True))
    if einzel_tbl.empty:
        st.info("Noch keine bestätigten Einzelmatches.")
    else:
        st.dataframe(einzel_tbl)

    # ---------- Letzte 5 Matches ----------
    st.subheader("Letzte 5 Matches")
    if matches.empty:
        st.info("Noch keine Spiele eingetragen.")
    else:
        recent = (
            matches.sort_values("Datum", ascending=False)
            .head(5)
            .reset_index(drop=True)
        )
        recent_display = recent.copy()
        recent_display["Datum"] = recent_display["Datum"].dt.strftime("%d.%m.%Y")
        st.dataframe(recent_display)
# endregion

# region Rundlauf Ansicht
# Rundlauf‑Ansicht
if st.session_state.view_mode == "round":
    st.title("Rundlauf eintragen")

    # Teilnehmer wählen
    selected = st.multiselect("Teilnehmer auswählen (mind. 3)", players["Name"])
    if len(selected) < 3:
        st.info("Mindestens drei Spieler auswählen.")
    else:
        # --- Finalisten‑Auswahl & Speichern nur wenn >=3 Teilnehmer ---
        fin_cols = st.columns(2)
        fin1 = fin_cols[0].selectbox("Finalist 1", selected, index=None, placeholder="Spieler wählen", key="r_f1")
        fin2 = fin_cols[1].selectbox("Finalist 2", [p for p in selected if p != fin1],
                                     index=None, placeholder="Spieler wählen", key="r_f2")
        sieger_options = [p for p in (fin1, fin2) if p is not None]
        sieger = st.selectbox("Sieger (muss Finalist sein)", sieger_options,
                              index=None, placeholder="Sieger wählen", key="r_win")

        save_disabled = (fin1 is None or fin2 is None or sieger is None)
        if st.button("Rundlauf speichern", disabled=save_disabled):
            new_row = {
                "Datum": datetime.now(ZoneInfo("Europe/Berlin")),
                "Teilnehmer": ";".join(selected),
                "Finalist1": fin1,
                "Finalist2": fin2,
                "Sieger": sieger,
                "creator": current_player,
                "confirmed_by": current_player  # Ersteller zählt sofort als bestätigt
            }
            pending_r = pd.concat([pending_r, pd.DataFrame([new_row])], ignore_index=True)
            save_csv(pending_r, PENDING_R)
            st.success("Rundlauf gespeichert – wartet auf Bestätigung eines Mitspielers.")

    # Alte 'conf'-Spalte endgültig entfernen, falls noch vorhanden
    if "conf" in pending_r.columns:
        pending_r.drop(columns=["conf"], inplace=True)
        save_csv(pending_r, PENDING_R)

    # Bestätigung
    with st.expander("Offene Rundläufe"):
        to_c = pending_r[
            (pending_r["Teilnehmer"].str.contains(current_player)) &
            (~pending_r["confirmed_by"].str.contains(current_player))
        ]
        if to_c.empty:
            st.info("Keine offenen Rundläufe für dich.")
        else:
            for idx,row in to_c.iterrows():
                st.write(f"Teilnehmer: {row['Teilnehmer']}  – Sieger: {row['Sieger']}")
                col1,col2 = st.columns(2)
                if col1.button("Bestätigen ✅", key=f"r_ok_{idx}"):
                    # Spieler in confirmed_by aufnehmen
                    confirmed = pending_r.at[idx,"confirmed_by"].split(";")
                    confirmed.append(current_player)
                    pending_r.at[idx,"confirmed_by"] = ";".join(confirmed)

                    # Finalisieren, wenn mind. 3 Bestätigende
                    if len(confirmed) >= 3:
                        rounds = pd.concat([rounds, pending_r.loc[[idx], rounds.columns]], ignore_index=True)
                        pending_r.drop(idx, inplace=True)
                        players = rebuild_players_r(players, rounds)
                        save_csv(rounds, ROUNDS)
                    save_csv(pending_r, PENDING_R)
                    save_csv(players, PLAYERS)
                    st.rerun()
                if col2.button("Ablehnen ❌", key=f"r_no_{idx}"):
                    pending_r.drop(idx,inplace=True)
                    save_csv(pending_r, PENDING_R)
                    st.rerun()

    st.subheader("Rundlauf‑Leaderboard")
    r_leader = (players[players["R_Spiele"] > 0][["Name","R_ELO","R_Siege","R_Zweite","R_Niederlagen","R_Spiele"]]
                .rename(columns={
                    "R_ELO":"ELO",
                    "R_Siege":"Siege",
                    "R_Zweite":"2. Platz",
                    "R_Niederlagen":"Niederlagen",
                    "R_Spiele":"Spiele"})
                .sort_values("ELO", ascending=False)
                .reset_index(drop=True))
    if r_leader.empty:
        st.info("Noch keine bestätigten Rundläufe.")
    else:
        st.dataframe(r_leader)

    # ---------- Letzte 5 Rundläufe ----------
    st.subheader("Letzte 5 Rundläufe")
    if rounds.empty:
        st.info("Noch keine Rundläufe eingetragen.")
    else:
        recent_r = (
            rounds.sort_values("Datum", ascending=False)
                  .head(5)
                  .reset_index(drop=True)
        )
        recent_r["Datum"] = recent_r["Datum"].dt.strftime("%d.%m.%Y")
        recent_r_display = recent_r[["Datum", "Sieger", "Finalist1", "Finalist2", "Teilnehmer"]]
        st.dataframe(recent_r_display)
    st.stop()
# endregion
