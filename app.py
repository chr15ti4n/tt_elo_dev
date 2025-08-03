import asyncio
import threading
import queue
from datetime import date, datetime
import time
import pandas as pd
import streamlit as st
from supabase import create_client, acreate_client

st.set_page_config(page_title="TT-ELO Realtime Demo", layout="wide")

# --- Debug-Helfer ---
if "debug_log" not in st.session_state:
    st.session_state.debug_log = []

def _dbg(msg: str):
    try:
        ts = datetime.now().strftime("%H:%M:%S")
        st.session_state.debug_log.append(f"[{ts}] {msg}")
        if len(st.session_state.debug_log) > 300:
            st.session_state.debug_log = st.session_state.debug_log[-300:]
    except Exception:
        pass

# Debug aktivierbar machen
try:
    debug_enabled = st.sidebar.checkbox("Debug anzeigen", value=True)
except Exception:
    debug_enabled = True

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
    _dbg(f"Initial load: {len(st.session_state.df)} rows")

# --- Gesehene IDs für Deduplizierung ---
if "seen_ids" not in st.session_state:
    try:
        st.session_state.seen_ids = set(
            st.session_state.df.get("id", pd.Series(dtype=str)).dropna().astype(str).tolist()
        )
    except Exception:
        st.session_state.seen_ids = set()
    _dbg(f"Init seen_ids: {len(st.session_state.seen_ids)}")

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
                .insert({"a": str(d)}, returning="representation")
                .execute()
            )
            _dbg("Insert executed")
            new_row = (res.data or [{"a": str(d)}])[0]
            _dbg(f"Insert response row id={new_row.get('id')}")

            # Lokal sofort anzeigen (nur wenn ID noch nicht vorhanden)
            row_id = str(new_row.get("id")) if isinstance(new_row, dict) else None
            if row_id and row_id not in st.session_state.seen_ids:
                st.session_state.df = pd.concat([pd.DataFrame([new_row]), st.session_state.df], ignore_index=True)
                st.session_state.seen_ids.add(row_id)
                _dbg(f"Locally appended row id={row_id}")
            st.success("Eintrag gespeichert.")
        except Exception as e:
            _dbg(f"Insert error: {e}")
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
        _dbg("Realtime subscription active (INSERT on tt_elo_events)")

        # Event-Loop offen halten
        while True:
            await asyncio.sleep(3600)

    asyncio.run(_run())


if not st.session_state.rt_started:
    t = threading.Thread(target=start_realtime_listener, daemon=True)
    t.start()
    st.session_state.rt_started = True

# neue Events abholen und inkrementell einfügen (kein kompletter Reload)
_dbg(f"Event queue size at start: ~{getattr(st.session_state.event_queue, 'qsize', lambda: 0)()}")
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
                row_id = str(row.get("id") or row.get("uuid") or "")
                _dbg(f"Queue payload processed: id={row_id or '?'}")
                if row_id and row_id not in st.session_state.seen_ids:
                    new_rows.append(row)
                    st.session_state.seen_ids.add(row_id)
        else:
            # unbekanntes Format einfach überspringen
            pass
    except queue.Empty:
        break

if new_rows:
    _dbg(f"Applying {len(new_rows)} new rows from realtime")
    try:
        new_df = pd.DataFrame(new_rows)
        # vorne anhängen, damit neueste oben stehen
        st.session_state.df = pd.concat([new_df, st.session_state.df], ignore_index=True)
        # nach inserted_at (falls vorhanden) absteigend sortieren
        if "inserted_at" in st.session_state.df.columns:
            try:
                st.session_state.df["inserted_at"] = pd.to_datetime(st.session_state.df["inserted_at"], errors="coerce")
                st.session_state.df = st.session_state.df.sort_values("inserted_at", ascending=False, na_position="last").reset_index(drop=True)
            except Exception:
                pass
        try:
            # sanfte Aktualisierung – falls add_rows nicht unterstützt ist, fallback auf komplettes Zeichnen
            table_elem.add_rows(new_df)
            _dbg("UI updated via add_rows")
        except Exception:
            _dbg("UI fallback redraw (container.dataframe)")
            container.dataframe(st.session_state.df, use_container_width=True, height=480)
    except Exception as e:
        st.warning(f"Konnte neue Zeilen nicht einfügen: {e}")

# Auto-Refresh: bevorzugt das Community-Paket, sonst Fallback mit rerun()
try:
    from streamlit_autorefresh import st_autorefresh
    _tick = st_autorefresh(interval=4000, key="refetch")
    _dbg(f"Auto-refresh tick={_tick}")
except Exception:
    _dbg("Autorefresh module missing – using fallback rerun in 4s")
    # Fallback ohne zusätzliches Paket (alle 4s neu rendern)
    time.sleep(4)
    try:
        st.rerun()
    except Exception:
        # ältere Streamlit-Versionen
        st.experimental_rerun()

# Periodische Resynchronisierung, falls ein Realtime-Event verpasst wurde
try:
    if isinstance(_tick, int) and _tick % 15 == 0:
        _dbg("Periodic resync started")
        resync = (
            supabase_sync
            .table(TABLE)
            .select("*")
            .order("inserted_at", desc=True)
            .execute()
        )
        fresh_df = pd.DataFrame(resync.data or [])
        _dbg(f"Resync fetched {len(fresh_df)} rows")
        if not fresh_df.empty:
            # Merge nach id
            if "id" in fresh_df.columns and "id" in st.session_state.df.columns:
                merged = (
                    pd.concat([fresh_df, st.session_state.df], ignore_index=True)
                    .drop_duplicates(subset=["id"], keep="first")
                )
            else:
                merged = pd.concat([fresh_df, st.session_state.df], ignore_index=True).drop_duplicates()
            _dbg(f"Resync merged → {len(merged)} rows total")
            if "inserted_at" in merged.columns:
                try:
                    merged["inserted_at"] = pd.to_datetime(merged["inserted_at"], errors="coerce")
                    merged = merged.sort_values("inserted_at", ascending=False, na_position="last").reset_index(drop=True)
                except Exception:
                    pass
            st.session_state.df = merged
            # Seen-IDs neu aufbauen
            try:
                st.session_state.seen_ids = set(merged.get("id", pd.Series(dtype=str)).dropna().astype(str).tolist())
            except Exception:
                pass
            container.dataframe(st.session_state.df, use_container_width=True, height=480)
except Exception:
    pass

# --- Debug-Panel ---
if debug_enabled:
    try:
        with st.sidebar.expander("Status / Metriken", expanded=True):
            st.metric("Rows", len(st.session_state.df))
            try:
                qsz = st.session_state.event_queue.qsize()
            except Exception:
                qsz = 0
            st.metric("Queue size", qsz)
            st.metric("Seen IDs", len(st.session_state.seen_ids))
            if not st.session_state.df.empty:
                preview = st.session_state.df[[c for c in ["id", "inserted_at", "a"] if c in st.session_state.df.columns]].head(3)
                st.write("Top rows:")
                st.dataframe(preview, use_container_width=True)
        with st.sidebar.expander("Debug Log", expanded=True):
            st.text("\n".join(st.session_state.debug_log[-120:]))
    except Exception:
        pass
