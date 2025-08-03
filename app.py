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

# --- Initialdaten einmalig laden und im State halten ---
if "df" not in st.session_state:
    try:
        res = (
            supabase_sync
            .table(TABLE)
            .select("*")
            .order("inserted_at", desc=True)
            .execute()
        )
        st.session_state.df = pd.DataFrame(res.data or [])
    except Exception as e:
        st.session_state.df = pd.DataFrame([])
        st.warning(f"Konnte Startdaten nicht laden: {e}")

# --- Formular zum Einfügen ---
with st.form("add_row", clear_on_submit=True):
    d = st.date_input("Datum (Spalte a)", value=date.today(), format="DD.MM.YYYY")
    submitted = st.form_submit_button("Eintrag speichern")
    if submitted:
        try:
            # In DB einfügen und die eingefügte Zeile zurückgeben
            res = (
                supabase_sync
                .table(TABLE)
                .insert({"a": str(d)})
                .select("*")
                .execute()
            )
            new_row = (res.data or [{"a": str(d)}])[0]

            # Lokal sofort anzeigen (vorne anfügen → neueste oben)
            st.session_state.df = pd.concat([pd.DataFrame([new_row]), st.session_state.df], ignore_index=True)
            st.success("Eintrag gespeichert.")
        except Exception as e:
            st.error(f"Konnte Eintrag nicht speichern: {e}")

# --- Tabelle rendern (stabil) ---
container = st.container()
try:
    table_elem = container.dataframe(
        st.session_state.df,
        use_container_width=True,
        height=480,
    )
except Exception as e:
    st.error(f"Fehler beim Rendern der Tabelle: {e}")

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

# neue Events abholen und inkrementell einfügen (kein kompletter Reload)
new_rows = []
while True:
    try:
        payload = st.session_state.event_queue.get_nowait()
        # Supabase-Postgres-Changes liefern i.d.R. payload["new"]
        if isinstance(payload, dict):
            row = (
                payload.get("new")
                or payload.get("record")
                or payload  # Fallback
            )
            if isinstance(row, dict):
                new_rows.append(row)
        else:
            # unbekanntes Format einfach überspringen
            pass
    except queue.Empty:
        break

if new_rows:
    try:
        new_df = pd.DataFrame(new_rows)
        # vorne anhängen, damit neueste oben stehen
        st.session_state.df = pd.concat([new_df, st.session_state.df], ignore_index=True)
        try:
            # sanfte Aktualisierung – falls add_rows nicht unterstützt ist, fallback auf komplettes Zeichnen
            table_elem.add_rows(new_df)
        except Exception:
            container.dataframe(st.session_state.df, use_container_width=True, height=480)
    except Exception as e:
        st.warning(f"Konnte neue Zeilen nicht einfügen: {e}")

# Auto-Refresh: bevorzugt das Community-Paket, sonst Fallback mit rerun()
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=4000, key="refetch")
except Exception:
    # Fallback ohne zusätzliches Paket (alle 4s neu rendern)
    time.sleep(4)
    try:
        st.rerun()
    except Exception:
        # ältere Streamlit-Versionen
        st.experimental_rerun()
        
