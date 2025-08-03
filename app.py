# app_dev.py
import os, asyncio, threading, datetime
import streamlit as st
from supabase import create_client

st.set_page_config(page_title="TT-ELO Realtime", layout="wide")

@st.cache_resource
def get_supabase():
    # 1) Streamlit secrets (Top-Level oder Abschnitt [supabase])
    url = None
    key = None

    if "SUPABASE_URL" in st.secrets:
        url = st.secrets["SUPABASE_URL"]
    elif "supabase" in st.secrets and "url" in st.secrets["supabase"]:
        url = st.secrets["supabase"]["url"]

    if "SUPABASE_ANON_KEY" in st.secrets:
        key = st.secrets["SUPABASE_ANON_KEY"]
    elif "supabase" in st.secrets:
        sub = st.secrets["supabase"]
        if "anon_key" in sub:
            key = sub["anon_key"]
        elif "key" in sub:
            key = sub["key"]

    # 2) Umgebungsvariablen als Fallback
    if not url:
        url = os.getenv("SUPABASE_URL")
    if not key:
        key = os.getenv("SUPABASE_ANON_KEY")

    # 3) .env als letzter Versuch (optional)
    if not url or not key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            url = url or os.getenv("SUPABASE_URL")
            key = key or os.getenv("SUPABASE_ANON_KEY")
        except Exception:
            pass

    if not url or not key:
        st.error(
            "Fehlende Supabase-Konfiguration. Lege entweder z. B. in .streamlit/secrets.toml\n"
            "A) Top-Level:\n  SUPABASE_URL=\"...\"\n  SUPABASE_ANON_KEY=\"...\"\n"
            "oder\n"
            "B) Abschnitt [supabase]:\n  [supabase]\n  url=\"...\"\n  key=\"...\"  # oder anon_key=\"...\"\n"
            "oder als Umgebungsvariablen SUPABASE_URL / SUPABASE_ANON_KEY an."
        )
        st.stop()

    return create_client(url, key)

# --- Auto-Login via Query-Params (user & token) ---

def init_user_from_query():
    q = st.query_params

    def _first(v):
        if isinstance(v, (list, tuple)):
            return v[0] if v else None
        return v

    auto_user = _first(q.get("user"))
    auto_token = _first(q.get("token"))

    if auto_user and auto_token:
        client = get_supabase()
        resp = client.table("players").select("pin").eq("name", auto_user).maybe_single().execute()
        data = resp.data
        row = None
        if isinstance(data, dict):
            row = data
        elif isinstance(data, list) and data:
            row = data[0]
        stored_hash = row.get("pin") if row else None
        if stored_hash and str(auto_token) == str(stored_hash):
            st.session_state["user"] = auto_user

# Initialisiere einmalig aus URL, falls noch kein User gesetzt ist
if "user" not in st.session_state:
    init_user_from_query()

def start_realtime_listener():
    try:
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
    except Exception as e:
        # notfalls im State hinterlegen
        st.session_state["rt_error"] = str(e)

if "rt_thread" not in st.session_state:
    t = threading.Thread(target=start_realtime_listener, daemon=True)
    t.start()
    st.session_state["rt_thread"] = True

st.title("🏓 TT-ELO – Live Leaderboard")
if "user" in st.session_state:
    st.caption(f"Angemeldet als: {st.session_state['user']}")

# leichte Auto-Refresh-Strategie (Fallback ohne st.autorefresh)
st.markdown("""
<meta http-equiv='refresh' content='3'>
""", unsafe_allow_html=True)

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
