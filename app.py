# region Imports
import streamlit as st
from supabase import create_client, Client
import pandas as pd
# endregion

# region Supabase Setup
try:
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
except KeyError:
    st.error("Bitte setze unter [supabase] url und key in deinen Streamlit Secrets.")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
# endregion

# region Authentication & CSV Display
if 'user' not in st.session_state:
    st.header("🔐 Login oder Registrierung")
    login_tab, register_tab = st.tabs(["Login", "Registrieren"])

    with login_tab:
        with st.form("login_form"):
            name = st.text_input("Name")
            pin  = st.text_input("PIN", type="password")
            if st.form_submit_button("Login"):
                # Simple in-memory check: try fetching this user
                resp = supabase.table("players").select("pin").eq("name", name).single().execute()
                if resp.data and resp.data.get("pin") == pin:
                    st.session_state.user = name
                    st.success(f"Eingeloggt als {name}")
                    st.rerun()
                else:
                    st.error("Ungültiger Name oder PIN")

    with register_tab:
        with st.form("register_form"):
            name = st.text_input("Neuer Benutzername")
            pin = st.text_input("PIN", type="password")
            pin_confirm = st.text_input("PIN wiederholen", type="password")
            if st.form_submit_button("Registrieren"):
                if not name:
                    st.error("Bitte einen Benutzernamen eingeben.")
                elif not pin:
                    st.error("Bitte eine PIN eingeben.")
                elif pin != pin_confirm:
                    st.error("PIN stimmt nicht überein.")
                else:
                    supabase.table("players").insert({"name": name, "pin": pin}).execute()
                    st.success("Registrierung erfolgreich. Bitte einloggen.")
    st.stop()
else:
    st.header(f"👋 Willkommen, {st.session_state.user}!")
    # Show the full players CSV from Supabase
    data = supabase.table("players").select("*").execute()
    df = pd.DataFrame(data.data)
    st.dataframe(df)
# endregion
