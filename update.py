"""
Neue Haupt-UI für TT-ELO (Demo-Build)
-------------------------------------
* Hauptmenü – Begrüßung, Gesamt-ELO, Unter-ELOs
* Drei Mini-Leaderboards (Einzel / Doppel / Rundlauf)
* Buttons: Einzel eingeben · Doppel eingeben · Rundlauf eingeben · Offene Matches bestätigen
* Sidebar: Aktualisieren · Regeln · Logout · Account löschen
* Zieht alle Hilfsfunktionen & Daten aus der bestehenden app.py
"""

import streamlit as st
from pathlib import Path
import app as core   # bestehende Logik wiederverwenden

# --- Daten laden --------------------------------------------------------------
players   = core.players
matches   = core.matches
doubles   = core.doubles
rounds    = core.rounds

# Gesamt-ELO (Gewichte nach Belieben anpassen)
W_EINZEL, W_DOPPEL, W_RUND = 0.6, 0.25, 0.15
players["G_ELO"] = (
    W_EINZEL * players["ELO"] +
    W_DOPPEL * players["D_ELO"] +
    W_RUND   * players["R_ELO"]
).round(0)

# --- Login-Check --------------------------------------------------------------
if not st.session_state.get("logged_in", False):
    st.warning("Bitte erst in der Haupt-App anmelden.")
    st.stop()

current_player = st.session_state["current_player"]

# --- Sidebar ------------------------------------------------------------------
with st.sidebar:
    if st.button("🔄 Aktualisieren", use_container_width=True):
        st.session_state["dfs"].clear()
        core._get_ws.cache_clear()
        st.rerun()

    if st.button("📜 Regeln", use_container_width=True):
        st.switch_page("app.py")            # Zeigt Regeln dort
    st.divider()
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.current_player = None
        st.query_params.clear()
        st.switch_page("app.py")
    if st.button("🗑️ Account löschen", use_container_width=True):
        st.switch_page("app.py")            # Admin-Funktion in Haupt-App

# --- Hauptseite ---------------------------------------------------------------
st.title("🏓 Tischtennis-Dashboard")

user_row = players.loc[players.Name == current_player].iloc[0]

st.markdown(f"### Willkommen, **{current_player}**!")
st.metric("Gesamt-ELO", int(user_row.G_ELO))

cols = st.columns(3)
cols[0].metric("Einzel-ELO",  int(user_row.ELO))
cols[1].metric("Doppel-ELO",  int(user_row.D_ELO))
cols[2].metric("Rundlauf-ELO", int(user_row.R_ELO))

st.divider()

def mini_lb(df, elo_col, title):
    st.subheader(title)
    tbl = (df.sort_values(elo_col, ascending=False)
             .loc[:, ["Name", elo_col]]
             .rename(columns={elo_col: "ELO"})
             .head(10)
             .reset_index(drop=True))
    st.dataframe(tbl, hide_index=True, width=350, height=240)

mini_lb(players[players.Spiele   > 0], "ELO",   "Einzel – Top 10")
mini_lb(players[players.D_Spiele > 0], "D_ELO", "Doppel – Top 10")
mini_lb(players[players.R_Spiele > 0], "R_ELO", "Rundlauf – Top 10")

st.divider()

btn = st.columns(4)
if btn[0].button("➕ Einzel eintragen"):
    st.switch_page("app.py")   # Einzel-Form dort
if btn[1].button("➕ Doppel eintragen"):
    st.switch_page("app.py")   # Doppel-Form
if btn[2].button("➕ Rundlauf eintragen"):
    st.switch_page("app.py")   # Rundlauf-Form
if btn[3].button("✅ Offene Matches bestätigen"):
    st.switch_page("app.py")   # Confirm-Expander

st.caption("Demo-Layout – alle Daten & Logik stammen aus der bestehenden app.py")
