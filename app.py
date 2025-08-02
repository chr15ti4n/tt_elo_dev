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
from datetime import datetime
from zoneinfo import ZoneInfo 
import bcrypt
# QR-Code generation
import qrcode
from pathlib import Path

from supabase import create_client

@st.cache_resource
def get_supabase_client():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)
supabase = get_supabase_client()

@st.cache_data
def load_table(table_name: str) -> pd.DataFrame:
    res = supabase.table(table_name).select("*").execute().data
    df = pd.DataFrame(res)
    # Normalize column names to lowercase for consistent access
    df.columns = [col.lower() for col in df.columns]
    # If table is empty, initialize with known schema columns for filtering
    if df.empty:
        schemas = {
            "matches":         ["id","datum","a","b","punktea","punkteb"],
            "pending_matches": ["id","datum","a","b","punktea","punkteb","confa","confb"],
            "doubles":         ["id","datum","a1","a2","b1","b2","punktea","punkteb"],
            "pending_doubles": ["id","datum","a1","a2","b1","b2","punktea","punkteb","confa","confb"],
            "rounds":          ["id","datum","teilnehmer","finalisten","sieger"],
            "pending_rounds":  ["id","datum","teilnehmer","finalisten","sieger","confa","confb"],
            "players":         ["id","name","elo","d_elo","r_elo","siege","niederlagen","spiele","pin","d_siege","d_niederlagen","d_spiele","r_siege","r_niederlagen","r_spiele","g_elo","r_zweite"]
        }
        cols = schemas.get(table_name, [])
        df = pd.DataFrame(columns=cols)
    if "datum" in df.columns:
        df["datum"] = pd.to_datetime(df["datum"], utc=True).dt.tz_convert("Europe/Berlin")
    return df



# ---------- QR-Code für Schnellzugriff ----------
QR_FILE = Path("form_qr.png")
APP_URL  = "https://tt-elo.streamlit.app"
if not QR_FILE.exists():
    qr_img = qrcode.make(APP_URL)
    qr_img.save(QR_FILE)



# region Helper Functions
# ---------- Hilfsfunktionen ----------

# --------- Session‑Cache für DataFrames (verhindert unnötige Sheets‑Reads) ---------

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
        a, b = row["a"], row["b"]
        pa, pb = int(row["punktea"]), int(row["punkteb"])

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
# Load normalized tables (columns all lowercase)
players   = load_table("players")
matches   = load_table("matches")
pending   = load_table("pending_matches")
pending_d = load_table("pending_doubles")
doubles   = load_table("doubles")
pending_r = load_table("pending_rounds")
rounds    = load_table("rounds")
# endregion

# region Doppel ELO Rebuild
# ---------- Doppel-Stats & ELO komplett neu berechnen ----------
def rebuild_players_d(players_df, doubles_df, k=48):
    players_df = players_df.copy()
    players_df[["D_ELO","D_Siege","D_Niederlagen","D_Spiele"]] = 0
    players_df["D_ELO"] = 1200
    for _, row in doubles_df.sort_values("Datum").iterrows():
        a1,a2,b1,b2 = row[["a1","a2","b1","b2"]]
        pa,pb = int(row["punktea"]), int(row["punkteb"])
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
        teilnehmer = row["teilnehmer"].split(";")
        fin1, fin2, winner = row["finalisten"].split(";")[0], row["finalisten"].split(";")[1], row["sieger"]
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
    """Rebuilds all ELO ratings after a confirmed match and upserts to Supabase."""
    global players, matches, doubles, rounds
    players = rebuild_players(players, matches)
    players = rebuild_players_d(players, doubles)
    players = rebuild_players_r(players, rounds)
    supabase.table("players").upsert(players.to_dict(orient="records"), on_conflict="id").execute()
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
        if auto_user in players["name"].values:
            stored_pin = players.loc[players["name"] == auto_user, "pin"].iat[0]
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
                if login_name not in players["name"].values:
                    st.error("Spielername nicht gefunden.")
                else:
                    stored_pin = players.loc[players["name"] == login_name, "pin"].iat[0]
                    if check_pin(login_pin, stored_pin):
                        # Falls PIN noch im Klartext war: sofort hash speichern
                        if not stored_pin.startswith("$2b$") and not stored_pin.startswith("$2a$"):
                            players.loc[players["name"] == login_name, "pin"] = hash_pin(login_pin)
                            supabase.table("players").update({"Pin": players.loc[players["name"] == login_name, "pin"].iat[0]}).eq("name", login_name).execute()
                        st.session_state.logged_in = True
                        st.session_state.current_player = login_name
                        # Save login in URL so refresh preserves session
                        st.query_params.update({
                            "user": login_name,
                            "token": players.loc[players["name"] == login_name, "pin"].iat[0],
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
            elif reg_name in players["name"].values:
                st.warning("Spieler existiert bereits.")
            else:
                new_player = {
                    "name": reg_name,
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
                supabase.table("players").insert([new_player]).execute()
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
                # Remove player from Supabase
                supabase.table("players").delete().eq("name", current_player).execute()
                # Remove matches, pending, doubles, pending_d, rounds, pending_r involving player
                supabase.table("matches").delete().or_(f"a.eq.{current_player},b.eq.{current_player}").execute()
                supabase.table("pending_matches").delete().or_(f"a.eq.{current_player},b.eq.{current_player}").execute()
                supabase.table("doubles").delete().or_(
                    f"a1.eq.{current_player},a2.eq.{current_player},b1.eq.{current_player},b2.eq.{current_player}"
                ).execute()
                supabase.table("pending_doubles").delete().or_(
                    f"a1.eq.{current_player},a2.eq.{current_player},b1.eq.{current_player},b2.eq.{current_player}"
                ).execute()
                # Remove from rounds and pending_rounds where Teilnehmer contains player
                if not rounds.empty:
                    for idx, row in rounds.iterrows():
                        if current_player in str(row["teilnehmer"]):
                            supabase.table("rounds").delete().eq("id", row["id"]).execute()
                if not pending_r.empty:
                    for idx, row in pending_r.iterrows():
                        if current_player in str(row["teilnehmer"]):
                            supabase.table("pending_rounds").delete().eq("id", row["id"]).execute()
                st.query_params.clear()  # clear URL params            
                st.session_state.logged_in = False
                st.session_state.current_player = None
                st.rerun()
        # Admin: vollständigen Rebuild auslösen
        if current_player in ADMINS:
            if st.button("🔄 Admin: Alle ELO neu berechnen", use_container_width=True):
                _rebuild_all()
                st.rerun()

if not st.session_state.logged_in:
    st.stop()
#
# endregion


current_player = st.session_state.current_player

# region Home Ansicht
if st.session_state.view_mode == "home":
    # CSS: Titel-Abstand reduzieren
    st.markdown(
        """
        <style>
        h1 {
            margin-top: 0rem !important;
            margin-bottom: 0rem !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        '<h1 style="text-align:center; margin-top:0rem; margin-bottom:0rem;">🏓 AK-Tischtennis</h1>',
        unsafe_allow_html=True
    )
    tab1, tab2, tab3 = st.tabs(["Willkommen", "Spielen", "Statistiken"])

    # Offene Matches für Bestätigung (auch vom Ersteller angezeigt)
    user = players.loc[players["name"] == current_player].iloc[0]
    # Einfache Bestätigung: pending solange confB False, für beide Teilnehmer
    sp = pending[
        ((pending["a"] == current_player) | (pending["b"] == current_player))
        & (~pending["confb"])
    ].copy()
    # Einfache Bestätigung: pending_d solange confB False, für beide Teams
    dp = pending_d[
        (
            (pending_d["a1"] == current_player) | (pending_d["a2"] == current_player)
            | (pending_d["b1"] == current_player) | (pending_d["b2"] == current_player)
        )
        & (~pending_d["confb"])
    ].copy()
    # Rundlauf: nur eine Bestätigung benötigt, Zeilen für Teilnehmer ohne Bestätigung
    rp = pending_r[
        pending_r["teilnehmer"].str.contains(current_player, na=False)
        & (~pending_r["confb"])
    ].copy()
    total_pending = len(sp) + len(dp) + len(rp)

    # Tab 1: Willkommen
    with tab1:
        st.markdown(
            f'<h3 style="text-align:center;">Willkommen, <strong>{current_player}</strong>!</h3>',
            unsafe_allow_html=True
        )
        # Kombiniere alle Modus-Matches modusunabhängig für Win-Streak
        combined = []
        # Einzelmatches
        df_s = matches[
        (matches["a"] == current_player) | (matches["b"] == current_player)
        ].copy()
        df_s["Win"] = df_s.apply(
        lambda r: (r["punktea"] > r["punkteb"]) if r["a"] == current_player
                  else (r["punkteb"] > r["punktea"]),
        axis=1
        )
        combined.append(df_s[["datum", "Win"]])
        # Doppelmatches
        df_d = doubles[
            (doubles["a1"] == current_player) | (doubles["a2"] == current_player)
            | (doubles["b1"] == current_player) | (doubles["b2"] == current_player)
        ].copy()
        df_d["Win"] = df_d.apply(
            lambda r: (r["punktea"] > r["punkteb"]) if current_player in (r["a1"], r["a2"])
                    else (r["punkteb"] > r["punktea"]),
            axis=1
        )
        combined.append(df_d[["datum", "Win"]])
        # Rundlaufmatches
        df_r = rounds[rounds["teilnehmer"].str.contains(current_player, na=False)].copy()
        df_r["Win"] = df_r["sieger"] == current_player
        combined.append(df_r[["datum", "Win"]])
        # Chronologisch sortieren
        comb_df = pd.concat(combined).sort_values("datum", ascending=False)

        # ELO-Übersicht optisch wie im Statistik-Tab
        st.markdown(
            f"""
            <div style="text-align:center; margin:1rem 0;">
                <div style="font-size:1.5rem; color:var(--text-secondary);">ELO</div>
                <div style="font-size:3rem; font-weight:bold; color:var(--text-primary);">{int(user["g_elo"])}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <style>
            [data-testid="stColumns"] {
                flex-wrap: nowrap !important;
                overflow-x: auto !important;
            }
            [data-testid="stColumn"] {
                min-width: 0 !important;
                flex: 1 1 auto !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        cols_e = st.columns(3)
        with cols_e[0]:
            st.markdown(
                f"""
                <div style="text-align:center;">
                    <div style="font-size:1.5rem; color:var(--text-secondary);">Einzel</div>
                    <div style="font-size:2.2rem; font-weight:bold; color:var(--text-primary);">{int(user["elo"])}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with cols_e[1]:
            st.markdown(
                f"""
                <div style="text-align:center;">
                    <div style="font-size:1.5rem; color:var(--text-secondary);">Doppel</div>
                    <div style="font-size:2.2rem; font-weight:bold; color:var(--text-primary);">{int(user["d_elo"])}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with cols_e[2]:
            st.markdown(
                f"""
                <div style="text-align:center;">
                    <div style="font-size:1.5rem; color:var(--text-secondary);">Rundlauf</div>
                    <div style="font-size:2.2rem; font-weight:bold; color:var(--text-primary);">{int(user["r_elo"])}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Win-Streak zentral als Einzeiler
            streak = 0
            for _, row in comb_df.iterrows():
                if row["Win"]:
                    streak += 1
                else:
                    break
            st.markdown(
                f"<div style='text-align:center; font-size:1.5rem; margin:1rem 0;'>Aktuelle Win-Streak: <strong>{streak}</strong> 🏆</div>",
                unsafe_allow_html=True
            )

            st.divider()
            cols_refresh1 = st.columns([4,1])
            with cols_refresh1[0]:
                st.subheader("Match-Bestätigungen")
            with cols_refresh1[1]:
                if st.button("🔄", key="refresh_tab1"):
                    if "dfs" in st.session_state:
                        st.session_state["dfs"].clear()
                    st.rerun()
            # Eingeladene Matches (nur diese können bestätigt werden)
            sp_inv = sp[sp["a"] != current_player]
            dp_inv = dp[~dp["a1"].eq(current_player) & ~dp["a2"].eq(current_player)]
            rp_inv = rp.copy()

            if sp_inv.empty and dp_inv.empty and rp_inv.empty:
                st.info("Keine offenen Matches.")
            # Einzel-Einladungen
            if not sp_inv.empty:
                    st.markdown("**Einzel**")
                    for idx, row in sp_inv.iterrows():
                        cols = st.columns([3,1,1])
                        cols[0].write(f"{row['A']} vs {row['B']}  {int(row['PunkteA'])}:{int(row['PunkteB'])}")
                        if cols[1].button("✅", key=f"confirm_s_{idx}"):
                            # Insert into Supabase matches
                            supabase.table("matches").insert([{
                                "Datum": row["Datum"], "A": row["A"], "B": row["B"],
                                "PunkteA": row["PunkteA"], "PunkteB": row["PunkteB"]
                            }]).execute()
                            _rebuild_all()
                            supabase.table("pending_matches").delete().eq("id", row["id"]).execute()
                            st.success("Match bestätigt! Bitte aktualisieren, um die Änderungen zu sehen.")
                            st.rerun()
                        if cols[2].button("❌", key=f"reject_s_{idx}"):
                            supabase.table("pending_matches").delete().eq("id", row["id"]).execute()
                            st.success("Match abgelehnt! Bitte aktualisieren, um die Änderungen zu sehen.")
                            st.rerun()

            # Doppel-Einladungen
            if not dp_inv.empty:
                st.markdown("**Doppel**")
                for idx, row in dp_inv.iterrows():
                    cols = st.columns([3,1,1])
                    cols[0].write(f"{row['A1']}/{row['A2']} vs {row['B1']}/{row['B2']}  {int(row['PunkteA'])}:{int(row['PunkteB'])}")
                    if cols[1].button("✅", key=f"confirm_d_{idx}"):
                        supabase.table("doubles").insert([{
                            "Datum": row["Datum"], "A1": row["A1"], "A2": row["A2"], "B1": row["B1"], "B2": row["B2"],
                            "PunkteA": row["PunkteA"], "PunkteB": row["PunkteB"]
                        }]).execute()
                        _rebuild_all()
                        supabase.table("pending_doubles").delete().eq("id", row["id"]).execute()
                        st.success("Match bestätigt! Bitte aktualisieren, um die Änderungen zu sehen.")
                        st.rerun()
                    if cols[2].button("❌", key=f"reject_d_{idx}"):
                        supabase.table("pending_doubles").delete().eq("id", row["id"]).execute()
                        st.success("Match abgelehnt! Bitte aktualisieren, um die Änderungen zu sehen.")
                        st.rerun()

            # Rundlauf-Einladungen
            if not rp_inv.empty:
                st.markdown("**Rundlauf**")
                for idx, row in rp_inv.iterrows():
                    cols = st.columns([3,1,1])
                    cols[0].write(f"{row['Teilnehmer']}  Sieger: {row['Sieger']}")
                    if cols[1].button("✅", key=f"confirm_r_{idx}"):
                        supabase.table("rounds").insert({
                            "Datum": row["Datum"],
                            "Teilnehmer": row["Teilnehmer"],
                            "Finalisten": row["Finalisten"],
                            "Sieger": row["Sieger"]
                        }).execute()
                        _rebuild_all()
                        supabase.table("pending_rounds").delete().eq("id", row["id"]).execute()
                        st.success("Match bestätigt! Bitte aktualisieren, um die Änderungen zu sehen.")
                        st.rerun()
                    if cols[2].button("❌", key=f"reject_r_{idx}"):
                        supabase.table("pending_rounds").delete().eq("id", row["id"]).execute()
                        st.success("Match abgelehnt! Bitte aktualisieren, um die Änderungen zu sehen.")
                        st.rerun()

            st.divider()
            # Allgemeine letzten 5 Matches (Update-Feed)
            df_sg = matches.copy()
            df_sg["Modus"] = "Einzel"
            df_sg["Teilnehmer"] = df_sg.apply(lambda r: f"{r['a']} vs {r['b']}", axis=1)
            df_sg["Ergebnis"] = df_sg.apply(lambda r: f"{int(r['punktea'])}:{int(r['punkteb'])}", axis=1)
            df_dg = doubles.copy()
            df_dg["Modus"] = "Doppel"
            df_dg["Teilnehmer"] = df_dg.apply(lambda r: f"{r['a1']}/{r['a2']} vs {r['b1']}/{r['b2']}", axis=1)
            df_dg["Ergebnis"] = df_dg.apply(lambda r: f"{int(r['punktea'])}:{int(r['punkteb'])}", axis=1)
            df_rg = rounds.copy()
            df_rg["Modus"] = "Rundlauf"
            df_rg["Teilnehmer"] = df_rg["teilnehmer"].str.replace(";", " / ")
            df_rg["Ergebnis"] = df_rg["sieger"]
            feed = pd.concat([df_sg[['datum','Modus','Teilnehmer','Ergebnis']],
                            df_dg[['datum','Modus','Teilnehmer','Ergebnis']],
                            df_rg[['datum','Modus','Teilnehmer','Ergebnis']]])
            feed = feed.sort_values("datum", ascending=False).head(5).reset_index(drop=True)
            st.subheader("Letzte Spiele")
            # Tabelle ohne Datum und Index
            feed_disp = feed[["Modus","Teilnehmer","Ergebnis"]]
            styler_feed = (
                feed_disp.style
                .set_properties(
                    subset=["Modus", "Ergebnis"],
                    **{"white-space": "nowrap", "min-width": "100px"}
                )
                .set_properties(
                    subset=["Teilnehmer"],
                    **{"white-space": "normal", "word-wrap": "break-word", "overflow-wrap": "anywhere"}
                )
                .set_table_styles([
                    {"selector": "th.row_heading, td.row_heading", "props": [("display", "none")]},
                    {"selector": "th.blank.level0", "props": [("display", "none")]}
                ])
            )
            st.table(styler_feed)

    # Tab 2: Match-Eintrag per Sub-Tabs und Bestätigung
    with tab2:
        # Unter-Tabs für Match-Eintrag
        mtab1, mtab2, mtab3 = st.tabs(["Einzel", "Doppel", "Rundlauf"])
        # Einzelmatch eintragen
        with mtab1:
            st.subheader("Eintrag Einzelmatch")
            date = st.date_input("Datum", value=datetime.now(ZoneInfo("Europe/Berlin")).date())
            dt = datetime.combine(date, datetime.min.time()).astimezone(ZoneInfo("Europe/Berlin"))
            opponent = st.selectbox("Gegner", [p for p in players["name"] if p != current_player])
            pts_a = st.number_input(f"Punkte {current_player}", min_value=0, max_value=100, value=11)
            pts_b = st.number_input(f"Punkte {opponent}", min_value=0, max_value=100, value=9)
            if st.button("Eintragen", key="einzel_submit"):
                supabase.table("pending_matches").insert([{
                    "Datum": dt.isoformat(), "A": current_player, "B": opponent,
                    "PunkteA": pts_a, "PunkteB": pts_b, "confA": True, "confB": False
                }]).execute()
                st.success("Einzel-Match erstellt! Bitte aktualisieren, um es zu sehen.")
                st.rerun()
        # Doppelmatch eintragen
        with mtab2:
            st.subheader("Eintrag Doppelmatch")
            date2 = st.date_input("Datum", value=datetime.now(ZoneInfo("Europe/Berlin")).date(), key="date_d")
            partner = st.selectbox("Partner", [p for p in players["name"] if p != current_player])
            dt2 = datetime.combine(date2, datetime.min.time()).astimezone(ZoneInfo("Europe/Berlin"))
            opp1 = st.selectbox("Gegner 1", [p for p in players["name"] if p not in [current_player, partner]])
            opp2 = st.selectbox("Gegner 2", [p for p in players["name"] if p not in [current_player, partner, opp1]])
            pts_ad = st.number_input("Punkte Team A", min_value=0, max_value=100, value=11)
            pts_bd = st.number_input("Punkte Team B", min_value=0, max_value=100, value=9)
            if st.button("Eintragen", key="doppel_submit"):
                supabase.table("pending_doubles").insert([{
                    "Datum": dt2.isoformat(), "A1": current_player, "A2": partner,
                    "B1": opp1, "B2": opp2, "PunkteA": pts_ad, "PunkteB": pts_bd, "confA": True, "confB": False
                }]).execute()
                st.success("Doppel-Match erstellt! Bitte aktualisieren, um es zu sehen.")
                st.rerun()
        # Rundlauf eintragen
        with mtab3:
            st.subheader("Eintrag Rundlauf")
            date3 = st.date_input("Datum", value=datetime.now(ZoneInfo("Europe/Berlin")).date(), key="date_r")
            dt3 = datetime.combine(date3, datetime.min.time()).astimezone(ZoneInfo("Europe/Berlin"))
            participants = st.multiselect("Teilnehmer", players["name"].tolist())
            finalists = st.multiselect("Finalisten (2)", participants, max_selections=2)
            winner = st.selectbox("Sieger", finalists, key="winner_r")
            if st.button("Eintragen", key="rund_submit"):
                part_str = ";".join(participants)
                f1, f2 = (finalists + ["", ""])[:2]
                supabase.table("pending_rounds").insert({
                    "Datum": dt3.isoformat(),
                    "Teilnehmer": part_str,
                    "Finalisten": f"{f1};{f2}",
                    "Sieger": winner,
                    "confA": True,
                    "confB": False
                }).execute()
                st.success("Rundlauf-Match erstellt! Bitte aktualisieren, um es zu sehen.")
                st.rerun()

        st.divider()
        cols_refresh2 = st.columns([4,1])
        with cols_refresh2[0]:
            st.subheader("Offene Matches")
        with cols_refresh2[1]:
            if st.button("🔄", key="refresh_tab2"):
                if "dfs" in st.session_state:
                    st.session_state["dfs"].clear()
                st.rerun()

        # Eingeladene Matches (Invitations)
        sp_inv = sp[sp["a"] != current_player]
        dp_inv = dp[~dp["a1"].eq(current_player) & ~dp["a2"].eq(current_player)]
        rp_inv = rp.copy()

        if sp_inv.empty and dp_inv.empty and rp_inv.empty:
            st.info("Keine ausstehenden Bestätigungen.")
        else:
            # Einzel-Einladungen
            if not sp_inv.empty:
                st.markdown("**Einzel**")
                for idx, row in sp_inv.iterrows():
                    cols = st.columns([3,1,1])
                    cols[0].write(f"{row['A']} vs {row['B']}  {int(row['PunkteA'])}:{int(row['PunkteB'])}")
                    if cols[1].button("✅", key=f"tab2_confirm_s_{idx}"):
                        supabase.table("matches").insert([{
                            "Datum": row["Datum"], "A": row["A"], "B": row["B"],
                            "PunkteA": row["PunkteA"], "PunkteB": row["PunkteB"]
                        }]).execute()
                        _rebuild_all()
                        supabase.table("pending_matches").delete().eq("id", row["id"]).execute()
                        st.success("Match bestätigt! Bitte aktualisieren, um die Änderungen zu sehen.")
                        st.rerun()
                    if cols[2].button("❌", key=f"tab2_reject_s_{idx}"):
                        supabase.table("pending_matches").delete().eq("id", row["id"]).execute()
                        st.success("Match abgelehnt! Bitte aktualisieren, um die Änderungen zu sehen.")
                        st.rerun()

            # Doppel-Einladungen
            if not dp_inv.empty:
                st.markdown("**Doppel**")
                for idx, row in dp_inv.iterrows():
                    cols = st.columns([3,1,1])
                    cols[0].write(f"{row['A1']}/{row['A2']} vs {row['B1']}/{row['B2']}  {int(row['PunkteA'])}:{int(row['PunkteB'])}")
                    if cols[1].button("✅", key=f"tab2_confirm_d_{idx}"):
                        supabase.table("doubles").insert([{
                            "Datum": row["Datum"], "A1": row["A1"], "A2": row["A2"], "B1": row["B1"], "B2": row["B2"],
                            "PunkteA": row["PunkteA"], "PunkteB": row["PunkteB"]
                        }]).execute()
                        _rebuild_all()
                        supabase.table("pending_doubles").delete().eq("id", row["id"]).execute()
                        st.success("Match bestätigt! Bitte aktualisieren, um die Änderungen zu sehen.")
                        st.rerun()
                    if cols[2].button("❌", key=f"tab2_reject_d_{idx}"):
                        supabase.table("pending_doubles").delete().eq("id", row["id"]).execute()
                        st.success("Match abgelehnt! Bitte aktualisieren, um die Änderungen zu sehen.")
                        st.rerun()
            # Rundlauf-Einladungen
            if not rp_inv.empty:
                st.markdown("**Rundlauf**")
                for idx, row in rp_inv.iterrows():
                    cols = st.columns([3,1,1])
                    cols[0].write(f"{row['Teilnehmer']}  Sieger: {row['Sieger']}")
                    if cols[1].button("✅", key=f"tab2_confirm_r_{idx}"):
                        supabase.table("rounds").insert({
                            "Datum": row["Datum"],
                            "Teilnehmer": row["Teilnehmer"],
                            "Finalisten": row["Finalisten"],
                            "Sieger": row["Sieger"]
                        }).execute()
                        _rebuild_all()
                        supabase.table("pending_rounds").delete().eq("id", row["id"]).execute()
                        st.success("Match bestätigt! Bitte aktualisieren, um die Änderungen zu sehen.")
                        st.rerun()
                    if cols[2].button("❌", key=f"tab2_reject_r_{idx}"):
                        supabase.table("pending_rounds").delete().eq("id", row["id"]).execute()
                        st.success("Match abgelehnt! Bitte aktualisieren, um die Änderungen zu sehen.")
                        st.rerun()

        # Eigene ausstehende Matches (Ersteller-Status)
        st.divider()
        st.subheader("Meine ausstehenden Matches")
        sp_cre = sp[sp["a"] == current_player]
        dp_cre = dp[(dp["a1"] == current_player) | (dp["a2"] == current_player)]
        rp_cre = pd.DataFrame(columns=rp.columns)

        if sp_cre.empty and dp_cre.empty and rp_cre.empty:
            st.info("Keine eigenen ausstehenden Matches.")
        else:
            if not sp_cre.empty:
                st.markdown("**Einzel**")
                for idx, row in sp_cre.iterrows():
                    cols = st.columns([3,1])
                    cols[0].write(f"{row['A']} vs {row['B']}  {int(row['PunkteA'])}:{int(row['PunkteB'])}")
                    if cols[1].button("❌", key=f"reject_own_s_{idx}"):
                        supabase.table("pending_matches").delete().eq("id", row["id"]).execute()
                        st.success("Match abgelehnt! Bitte aktualisieren, um die Änderungen zu sehen.")
                        st.rerun()

            if not dp_cre.empty:
                st.markdown("**Doppel**")
                for idx, row in dp_cre.iterrows():
                    cols = st.columns([3,1])
                    cols[0].write(f"{row['A1']}/{row['A2']} vs {row['B1']}/{row['B2']}  {int(row['PunkteA'])}:{int(row['PunkteB'])}")
                    if cols[1].button("❌", key=f"reject_own_d_{idx}"):
                        supabase.table("pending_doubles").delete().eq("id", row["id"]).execute()
                        st.success("Match abgelehnt! Bitte aktualisieren, um die Änderungen zu sehen.")
                        st.rerun()

            if not rp_cre.empty:
                st.markdown("**Rundlauf**")
                for idx, row in rp_cre.iterrows():
                    cols = st.columns([3,1])
                    cols[0].write(f"{row['Teilnehmer']}  Sieger: {row['Sieger']}")
                    if cols[1].button("❌", key=f"reject_own_r_{idx}"):
                        supabase.table("pending_rounds").delete().eq("id", row["id"]).execute()
                        st.success("Match abgelehnt! Bitte aktualisieren, um die Änderungen zu sehen.")
                        st.rerun()

    # Tab 3: Leaderboards und Statistiken (wie bisher)
    with tab3:
        # Kombiniere alle Modus-Matches modusunabhängig (für Win-Streak und letzte 5 Matches)
        combined = []
        # Einzelmatches
        df_s = matches[
            (matches["a"] == current_player) | (matches["b"] == current_player)
        ].copy()
        df_s["Win"] = df_s.apply(
            lambda r: (r["punktea"] > r["punkteb"]) if r["a"] == current_player
                      else (r["punkteb"] > r["punktea"]),
            axis=1
        )
        combined.append(df_s[["datum", "Win"]])
        # Doppelmatches
        df_d = doubles[
            (doubles["a1"] == current_player) | (doubles["a2"] == current_player)
            | (doubles["b1"] == current_player) | (doubles["b2"] == current_player)
        ].copy()
        df_d["Win"] = df_d.apply(
            lambda r: (r["punktea"] > r["punkteb"]) if current_player in (r["a1"], r["a2"])
                      else (r["punkteb"] > r["punktea"]),
            axis=1
        )
        combined.append(df_d[["datum", "Win"]])
        # Rundlaufmatches
        df_r = rounds[rounds["teilnehmer"].str.contains(current_player, na=False)].copy()
        df_r["Win"] = df_r["sieger"] == current_player
        combined.append(df_r[["datum", "Win"]])
        # Chronologisch sortieren
        comb_df = pd.concat(combined).sort_values("datum", ascending=False)

        st.markdown(
            """
            <style>
            .stTable table tr th:first-child,
            .stTable table tr td:first-child {
                display: none;
            }
            /* Center Streamlit tables */
            .stTable, .stTable table {
                margin-left: auto !important;
                margin-right: auto !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        # ELO-Übersicht optisch wie im Dashboard
        st.markdown(
            f"""
            <div style="text-align:center; margin:1rem 0;">
              <div style="font-size:1.5rem; color:var(--text-secondary);">ELO</div>
              <div style="font-size:3rem; font-weight:bold; color:var(--text-primary);">{int(user["g_elo"])}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        # CSS: Verhindere Umbruch der ELO-Modus-Spalten auf Mobil
        st.markdown(
            """
            <style>
            [data-testid="stColumns"] {
                flex-wrap: nowrap !important;
                overflow-x: auto !important;
            }
            [data-testid="stColumn"] {
                min-width: 0 !important;
                flex: 1 1 auto !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        cols = st.columns(3)
        with cols[0]:
            st.markdown(
                f"""
                <div style="text-align:center;">
                  <div style="font-size:1.5rem; color:var(--text-secondary);">Einzel</div>
                  <div style="font-size:2.2rem; font-weight:bold; color:var(--text-primary);">{int(user["elo"])}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with cols[1]:
            st.markdown(
                f"""
                <div style="text-align:center;">
                  <div style="font-size:1.5rem; color:var(--text-secondary);">Doppel</div>
                  <div style="font-size:2.2rem; font-weight:bold; color:var(--text-primary);">{int(user["d_elo"])}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with cols[2]:
            st.markdown(
                f"""
                <div style="text-align:center;">
                  <div style="font-size:1.5rem; color:var(--text-secondary);">Rundlauf</div>
                  <div style="font-size:2.2rem; font-weight:bold; color:var(--text-primary);">{int(user["r_elo"])}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        # Aktuelle Win-Streak als Einzeiler
        streak = 0
        for _, row in comb_df.iterrows():
            if row["Win"]:
                streak += 1
            else:
                break
        # Win-Streak zentral als Einzeiler mit einheitlicher Schriftgröße, outside columns for centering
        st.markdown(
            f"<div style='text-align:center; font-size:1.5rem; margin:1rem 0;'>Aktuelle Win-Streak: <strong>{streak}</strong> 🏆</div>",
            unsafe_allow_html=True
        )

        st.subheader("Leaderboard")
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
        for idx, metric in enumerate(["g_elo", "elo", "d_elo", "r_elo"]):
            with sub_tabs[idx]:
                # Tabelle nur mit Name und ausgewählter Elo-Spalte
                df_tab = players[["name", metric]].copy()
                df_tab = df_tab.rename(columns={"name": "Name", metric: "ELO"})
                df_tab["ELO"] = df_tab["ELO"].astype(int)
                # Absteigend sortieren
                df_tab = df_tab.sort_values("ELO", ascending=False).reset_index(drop=True)
                # Highlight aktuellen Spieler
                def highlight_current(row):
                    return [
                        "background-color: #ADD8E6; color: black"
                        if row["Name"] == current_player else ""
                        for _ in row
                    ]
                styler = df_tab.style.apply(highlight_current, axis=1)
                # Index-Spalte ausblenden
                styler = styler.set_table_styles([
                    {"selector": "th.row_heading, td.row_heading", "props": [("display", "none")]},
                    {"selector": "th.blank.level0", "props": [("display", "none")]}
                ])
                st.table(styler)

        # Persönliche Statistiken in kompakter Tabelle (Kategorie, Spiele, Siege, Winrate %)
        st.markdown("---")
        st.subheader("Meine Statistiken")
        # DataFrame mit persönlichen Werten
        stats = pd.DataFrame({
            "Kategorie": ["Einzel", "Doppel", "Rundlauf", "Gesamt"],
            "Spiele": [
                int(user["spiele"]),
                int(user["d_spiele"]),
                int(user["r_spiele"]),
                int(user["spiele"] + user["d_spiele"] + user["r_spiele"])
            ],
            "Siege": [
                int(user["siege"]),
                int(user["d_siege"]),
                int(user["r_siege"]),
                int(user["siege"] + user["d_siege"] + user["r_siege"])
            ]
        })
        # Winrate (%) in Prozent berechnen (ganzzahlig)
        stats["Winrate (%)"] = stats.apply(
            lambda row: int(round((row["Siege"] / row["Spiele"]) * 100))
            if row["Spiele"] > 0 else 0,
            axis=1
        )

        # Statische Tabelle ohne Index, centered
        st.table(stats)

        # Darstellung der letzten 5 Matches modusunabhängig
        combined = []
        # Einzelmatches
        df_s = matches[
            (matches["a"] == current_player) | (matches["b"] == current_player)
        ].copy()
        df_s["Modus"] = "Einzel"
        df_s["Gegner"] = df_s.apply(
            lambda r: r["b"] if r["a"] == current_player else r["a"], axis=1
        )
        df_s["Ergebnis"] = df_s.apply(
            lambda r: f"{int(r['punktea'])}:{int(r['punkteb'])}" if r["a"] == current_player
                      else f"{int(r['punkteb'])}:{int(r['punktea'])}",
            axis=1
        )
        df_s["Win"] = df_s.apply(
            lambda r: (r["punktea"] > r["punkteb"]) if r["a"] == current_player
                      else (r["punkteb"] > r["punktea"]),
            axis=1
        )
        combined.append(df_s[["datum", "Modus", "Gegner", "Ergebnis", "Win"]])

        # Doppelmatches
        df_d = doubles[
            (doubles["a1"] == current_player) | (doubles["a2"] == current_player)
            | (doubles["b1"] == current_player) | (doubles["b2"] == current_player)
        ].copy()
        df_d["Modus"] = "Doppel"
        df_d["Gegner"] = df_d.apply(
            lambda r: f"{r['b1']} / {r['b2']}" if current_player in (r["a1"], r["a2"])
                      else f"{r['a1']} / {r['a2']}",
            axis=1
        )
        df_d["Ergebnis"] = df_d.apply(
            lambda r: f"{int(r['punktea'])}:{int(r['punkteb'])}" if current_player in (r["a1"], r["a2"])
                      else f"{int(r['punkteb'])}:{int(r['punktea'])}",
            axis=1
        )
        df_d["Win"] = df_d.apply(
            lambda r: (r["punktea"] > r["punkteb"]) if current_player in (r["a1"], r["a2"])
                      else (r["punkteb"] > r["punktea"]),
            axis=1
        )
        combined.append(df_d[["datum", "Modus", "Gegner", "Ergebnis", "Win"]])

        # Rundlaufmatches
        df_r = rounds[rounds["teilnehmer"].str.contains(current_player, na=False)].copy()
        df_r["Modus"] = "Rundlauf"
        df_r["Gegner"] = df_r["teilnehmer"].str.replace(";", " / ")
        df_r["Ergebnis"] = df_r["sieger"]
        df_r["Win"] = df_r["sieger"] == current_player
        combined.append(df_r[["datum", "Modus", "Gegner", "Ergebnis", "Win"]])

        # Vereinigen, sortieren und letzte 5 anzeigen
        comb_df = pd.concat(combined).sort_values("datum", ascending=False)
        last5 = comb_df.head(5).reset_index(drop=True)
        st.subheader("Meine letzten 5 Spiele")
        last5_disp = last5[["Modus", "Gegner", "Ergebnis"]].reset_index(drop=True)
        styler_last5 = (
            last5_disp.style
            .set_properties(
                subset=["Modus", "Ergebnis"],
                **{"white-space": "nowrap", "min-width": "100px"}
            )
            .set_properties(
                subset=["Gegner"],
                **{"white-space": "normal", "word-wrap": "break-word", "overflow-wrap": "anywhere"}
            )
            .set_table_styles([
                {"selector": "th.row_heading, td.row_heading", "props": [("display", "none")]},
                {"selector": "th.blank.level0", "props": [("display", "none")]}
            ])
        )
        st.table(styler_last5)

        # Modal: Offene Matches bestätigen
        if st.session_state.show_confirm_modal:
            with ui_container("Offene Matches bestätigen"):
                if total_pending == 0:
                    st.info("Keine offenen Matches.")
                else:
                    if not sp.empty:
                        st.subheader("Einzel-Matches")
                        st.table(sp[["datum", "a", "punktea", "punkteb", "b"]])
                    if not dp.empty:
                        st.subheader("Doppel-Matches")
                        st.table(dp[["datum", "a1", "a2", "b1", "b2", "punktea", "punkteb"]])
                    if not rp.empty:
                        st.subheader("Rundlauf-Matches")
                        st.table(rp[["datum", "teilnehmer", "sieger"]])
                # Close button
                if st.button("Schließen"):
                    st.session_state.show_confirm_modal = False
                    st.rerun()

    st.stop()
# endregion
 
