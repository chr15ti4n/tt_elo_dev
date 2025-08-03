

import asyncio
import threading
import queue
from datetime import date
import time
import pandas as pd
import streamlit as st
from supabase import create_client, acreate_client

st.set_page_config(page_title="TT-ELO Realtime Demo", layout="wide")

# --- Supabase config aus Streamlit-Secrets ---
# Erwartetes Format in .streamlit/secrets.toml:
# [supabase]
# url = "..."
# key = "..."
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["key"]

TABLE = "tt_elo_events"

# Sync-Client für normale Abfragen/Insert
supabase_sync = create_client(url, key)

st.title("TT-ELO – Datumseinträge (Realtime)")

# --- Formular zum Einfügen ---
with st.form("add_row", clear_on_submit=True):
    d = st.date_input("Datum (Spalte a)", value=date.today(), format="DD.MM.YYYY")
    submitted = st.form_submit_button("Eintrag speichern")
    if submitted:
        # Datum als ISO-String an Postgres schicken
        supabase_sync.table(TABLE).insert({"a": str(d)}).execute()
        st.success("Eintrag gespeichert.")

# --- Daten laden & anzeigen ---

def fetch_df():
    res = (
        supabase_sync
        .table(TABLE)
        .select("*")
        .order("inserted_at", desc=True)
        .execute()
    )
    return pd.DataFrame(res.data or [])

data_placeholder = st.empty()
try:
    data_placeholder.dataframe(fetch_df(), use_container_width=True)
except Exception as e:
    st.error(f"Fehler beim Laden der Daten: {e}")

# --- Realtime-Listener (Async) ---
# Hinweis: Realtime ist im Python-Client nur im ASYNC-Client verfügbar.
# Wir starten daher eine Hintergrund-Thread mit einer asyncio-Loop,
# abonnieren Postgres INSERT-Events und legen eingehende Events in eine Queue.

if "rt_started" not in st.session_state:
    st.session_state.rt_started = False
if "event_queue" not in st.session_state:
    st.session_state.event_queue = queue.Queue()


def start_realtime_listener():
    async def _run():
        # Async-Client für Realtime
        client = await acreate_client(url, key)
        channel = client.channel("room_tt_elo")

        def on_insert(payload):
            # Neues Event in Queue legen (thread-safe)
            try:
                st.session_state.event_queue.put_nowait(payload)
            except Exception:
                pass

        await (
            channel
            .on_postgres_changes(
                event="INSERT", schema="public", table=TABLE, callback=on_insert
            )
            .subscribe()
        )

        # Event-Loop offen halten
        while True:
            await asyncio.sleep(3600)

    asyncio.run(_run())


if not st.session_state.rt_started:
    t = threading.Thread(target=start_realtime_listener, daemon=True)
    t.start()
    st.session_state.rt_started = True

# neue Events abholen und bei Bedarf neu laden
new_events = 0
while True:
    try:
        _ = st.session_state.event_queue.get_nowait()
        new_events += 1
    except queue.Empty:
        break

if new_events:
    st.toast(f"{new_events} neuer Eintrag angekommen")
    try:
        data_placeholder.dataframe(fetch_df(), use_container_width=True)
    except Exception as e:
        st.error(f"Fehler beim Nachladen der Daten: {e}")

# Auto-Refresh: bevorzugt das Community-Paket, sonst Fallback mit rerun()
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=2000, key="refetch")
except Exception:
    # Fallback ohne zusätzliches Paket (alle 2s neu rendern)
    time.sleep(2)
    try:
        st.rerun()
    except Exception:
        # ältere Streamlit-Versionen
        st.experimental_rerun()
