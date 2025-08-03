# app_dev.py
import asyncio, threading, datetime
import streamlit as st
from supabase import create_client

st.set_page_config(page_title="TT-ELO Realtime", layout="wide")

@st.cache_resource
def get_supabase():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"])

# --- Auto-Login via Query-Params (user & token) ---

def init_user_from_query():
    q = st.query_params
    auto_user = q.get("user")
    auto_token = q.get("token")
    if auto_user and auto_token:
        client = get_supabase()
        resp = client.table("players").select("pin").eq("name", auto_user).maybe_single().execute()
        stored_hash = resp.data["pin"] if resp.data else None
        if stored_hash and auto_token == stored_hash:
            st.session_state["user"] = auto_user

# Initialisiere einmalig aus URL, falls noch kein User gesetzt ist
if "user" not in st.session_state:
    init_user_from_query()

def start_realtime_listener():
    sb = get_supabase()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    channel = sb.channel("rt-matches")

    def handle_change(payload):
        # bump a counter to trigger UI refresh
        st.session_state["rt_bump"] = st.session_state.get("rt_bump", 0) + 1

    async def subscribe():
        await channel.on_postgres_changes(
            event="*",
            schema="public",
            table="matches",
            callback=handle_change,
        ).subscribe()

    loop.run_until_complete(subscribe())
    loop.run_forever()

if "rt_thread" not in st.session_state:
    t = threading.Thread(target=start_realtime_listener, daemon=True)
    t.start()
    st.session_state["rt_thread"] = True

st.title("🏓 TT-ELO – Live Leaderboard")
if "user" in st.session_state:
    st.caption(f"Angemeldet als: {st.session_state['user']}")

# leichte Auto-Refresh-Strategie
st.autorefresh(interval=3000, key="tick", limit=None)

sb = get_supabase()
leaderboard = sb.table("v_leaderboard").select("*").execute().data
st.dataframe(leaderboard, use_container_width=True)

# Beispiel: neues Match anlegen (Test)
with st.form("new_match"):
    col1, col2, col3, col4 = st.columns(4)
    a = col1.text_input("Player A ID")
    b = col2.text_input("Player B ID")
    sa = col3.number_input("Score A", 0, 100, 3)
    sb_ = col4.number_input("Score B", 0, 100, 1)
    if st.form_submit_button("Match speichern"):
        sb.table("matches").insert({"player_a": a, "player_b": b, "score_a": int(sa), "score_b": int(sb_)}).execute()
        st.success("Match gespeichert – sollte realtime erscheinen.")
