# region import
import streamlit as st
from supabase import create_client, Client
import pandas as pd
from datetime import datetime
# endregion

# region Load Streamlit Secrets
try:
    SUPABASE_URL: str = st.secrets["supabase"]["url"]
    SUPABASE_KEY: str = st.secrets["supabase"]["key"]
except KeyError:
    st.error("Bitte setze unter [supabase] url und key in deinen Streamlit Secrets.")
    st.stop()
# endregion

# region Supabase Client Initialization
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
# endregion

# region Helper Functions
@st.cache_data(show_spinner=False)
def load_table(table_name: str) -> pd.DataFrame:
    data = supabase.table(table_name).select("*").order("created_at", desc=True).execute()
    return pd.DataFrame(data.data)
# endregion

# region CRUD Functions
def add_player(name: str, pin: str):
    supabase.table("players").insert({"name": name, "pin": pin}).execute()

def add_match(a: str, b: str, punktea: int, punkteb: int):
    supabase.table("matches").insert({
        "datum": datetime.utcnow().isoformat(),
        "a": a, "b": b,
        "punktea": punktea, "punkteb": punkteb
    }).execute()
# endregion

# region Streamlit Layout
st.title("🏓 Tischtennis Elo App")

# region Sidebar Navigation
menu = st.sidebar.selectbox("Navigation", ["Spieler", "Einzelmatch", "Leaderboard"])
# endregion

# region Spieler Section
if menu == "Spieler":
    st.header("Spieler hinzufügen")
    with st.form("player_form", clear_on_submit=True):
        name = st.text_input("Name")
        pin  = st.text_input("PIN/Passwort", type="password")
        submitted = st.form_submit_button("Hinzufügen")
        if submitted:
            if name and pin:
                add_player(name, pin)
                st.success(f"Spieler {name} hinzugefügt.")
            else:
                st.error("Bitte Namen und PIN angeben.")

    st.header("Alle Spieler")
    df_players = load_table("players")
    st.dataframe(df_players[["name", "elo", "spiele", "siege", "niederlagen"]])
# endregion

# region Einzelmatch Section
elif menu == "Einzelmatch":
    st.header("Einzelmatch eintragen")
    df_players = load_table("players")
    names = df_players["name"].tolist()
    with st.form("match_form", clear_on_submit=True):
        a = st.selectbox("Spieler A", names)
        b = st.selectbox("Spieler B", names, index=1)
        punktea = st.number_input("Punkte A", min_value=0, step=1)
        punkteb = st.number_input("Punkte B", min_value=0, step=1)
        ok = st.form_submit_button("Einreichen")
        if ok:
            if a != b:
                add_match(a, b, punktea, punkteb)
                st.success("Match eingetragen.")
            else:
                st.error("Spieler A und B müssen unterschiedlich sein.")

    st.header("Letzte Matches")
    df_matches = load_table("matches")
    st.dataframe(df_matches[["datum", "a", "b", "punktea", "punkteb"]])
# endregion

# region Leaderboard Section
elif menu == "Leaderboard":
    st.header("Aktuelle Elo-Rangliste")
    df = load_table("players")
    df = df.sort_values("elo", ascending=False)
    st.dataframe(df[["name", "elo", "spiele", "siege", "niederlagen"]])
# endregion
# endregion

# region Footer
st.markdown("---")
st.caption("Elo-App für Einzel, Doppel und Rundlauf – entwickle sie Stück für Stück weiter!")
# endregion
