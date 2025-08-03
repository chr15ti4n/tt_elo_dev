# region Imports
import streamlit as st
from supabase import create_client, Client
import pandas as pd
import bcrypt
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


import time
import asyncio
import threading
import importlib

from queue import SimpleQueue
import traceback
import inspect

# Async Realtime import (must be early so availability shows correctly)
try:
    from supabase import acreate_client, AsyncClient  # async client for realtime
except Exception:
    acreate_client = None
    AsyncClient = None

# Thread-safe comms between realtime thread and main thread
_RT_QUEUE = SimpleQueue()
_RT_FLAGS = {
    "client_created": False,
    "channel_subscribed": False,
    "connect_ok": False,
    "last_error": None,
}

# region UI Helpers (editing state)
# region UI Helpers (editing state)
def _set_editing_true():
    st.session_state["editing"] = True
def _set_editing_false():
    st.session_state["editing"] = False

# region Recent combined (Last games)
@st.cache_data(ttl=5)
def load_last_events_table(limit: int = 5) -> pd.DataFrame:
    # Fetch recent items per mode
    m = supabase.table("matches").select("datum,a,b,punktea,punkteb").order("datum", desc=True).limit(limit).execute().data or []
    d = supabase.table("doubles").select("datum,a1,a2,b1,b2,punktea,punkteb").order("datum", desc=True).limit(limit).execute().data or []
    r = supabase.table("rounds").select("datum,teilnehmer,finalisten,sieger").order("datum", desc=True).limit(limit).execute().data or []

    rows = []
    for x in m:
        rows.append({
            "Modus": "Einzel",
            "Teilnehmer": f"{x['a']} vs {x['b']}",
            "Ergebnis": f"{x['punktea']}:{x['punkteb']}",
            "datum": x.get("datum"),
        })
    for x in d:
        rows.append({
            "Modus": "Doppel",
            "Teilnehmer": f"{x['a1']}+{x['a2']} vs {x['b1']}+{x['b2']}",
            "Ergebnis": f"{x['punktea']}:{x['punkteb']}",
            "datum": x.get("datum"),
        })
    for x in r:
        parts = ", ".join([t for t in (x.get('teilnehmer','').split(';') if x.get('teilnehmer') else []) if t])
        finals = [f for f in (x.get('finalisten','').split(';') if x.get('finalisten') else []) if f]
        winner = x.get('sieger','')
        second = ""
        if len(finals) >= 2:
            if finals[0] == winner:
                second = finals[1]
            elif finals[1] == winner:
                second = finals[0]
            else:
                # Fallback: wenn Sieger nicht in den Finalisten steht, nimm den ersten als Zweiten
                second = finals[0]
        res = f"1. {winner}\n2. {second}" if second else f"1. {winner}"
        rows.append({
            "Modus": "Rundlauf",
            "Teilnehmer": parts,
            "Ergebnis": res,
            "datum": x.get("datum"),
        })

    # Sort by datetime desc across modes and clip to limit
    rows = sorted(rows, key=lambda r: r.get("datum") or "", reverse=True)[:limit]
    if rows:
        df = pd.DataFrame(rows, columns=["Modus","Teilnehmer","Ergebnis","datum"]) 
        return df[["Modus","Teilnehmer","Ergebnis"]]
    else:
        return pd.DataFrame(columns=["Modus","Teilnehmer","Ergebnis"])
# endregion

# region Time Helpers (TZ)
BERLIN = ZoneInfo("Europe/Berlin")
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def fmt_dt_local(val) -> str:
    try:
        dt = pd.to_datetime(val, utc=True)
        if pd.isna(dt):
            return ""
        return dt.tz_convert(BERLIN).strftime("%d.%m.%Y %H:%M")
    except Exception:
        return str(val)
# endregion

# region Supabase client version
try:
    supa_mod = importlib.import_module("supabase")
    SUPABASE_PY_VERSION = getattr(supa_mod, "__version__", "unknown")
except Exception:
    SUPABASE_PY_VERSION = "unknown"
# endregion
# endregion

# region PIN Hashing
def hash_pin(pin: str) -> str:
    """Generates a bcrypt hash for the provided PIN."""
    return bcrypt.hashpw(pin.encode(), bcrypt.gensalt()).decode()

def check_pin(pin: str, stored: str) -> bool:
    """
    Compares the entered PIN with the stored value.
    Supports both legacy plaintext and bcrypt hashes.
    """
    if stored.startswith("$2b$") or stored.startswith("$2a$"):
        return bcrypt.checkpw(pin.encode(), stored.encode())
    else:
        return pin == stored
# endregion

# region Supabase Setup
try:
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
except KeyError:
    st.error("Bitte setze unter [supabase] url und key in deinen Streamlit Secrets.")
    st.stop()


# Initialize realtime debug baseline (always visible)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
st.session_state.setdefault("_rt_debug", {})
st.session_state["_rt_debug"].update({
    "supabase_py_version": SUPABASE_PY_VERSION,
    "acreate_client_available": bool(acreate_client is not None),
    "acreate_is_coro": bool(acreate_client is not None and inspect.iscoroutinefunction(acreate_client)),
    "async_client_available": bool(AsyncClient is not None),
    "phase": "init",
})
# endregion


# region Supabase Realtime (event-driven refresh)

# Smoketest for Realtime
async def _realtime_smoketest() -> tuple[bool, str]:
    """Try to create async client, connect, subscribe, then disconnect.
    Returns (ok, message)."""
    try:
        if acreate_client is None:
            return False, "acreate_client ist nicht verfügbar (supabase-Paket zu alt?)."
        acli = await acreate_client(SUPABASE_URL, SUPABASE_KEY)
        await acli.realtime.connect()
        chan = acli.channel("smoketest")
        await chan.subscribe()
        # brief no-op delay to allow handshake
        try:
            await asyncio.sleep(0.2)
        except Exception:
            pass
        # clean up
        await acli.realtime.disconnect()
        return True, "Realtime-Verbindung (connect + subscribe) erfolgreich."
    except Exception as e:
        return False, f"Smoketest-Fehler: {type(e).__name__}: {e}"

def _ensure_realtime_started():
    """Start a background task that subscribes to DB changes and flips a flag using a background thread.
    Requires: Realtime enabled for the tables in Supabase (Database → Replication → supabase_realtime).
    """
    # If a previous worker thread exists and is alive, keep it
    t = st.session_state.get("_rt_thread")
    if t and getattr(t, "is_alive", lambda: False)():
        return

    if acreate_client is None:
        # record why realtime did not start
        st.session_state.setdefault("_rt_debug", {})
        st.session_state["_rt_debug"]["last_error"] = (
            f"Realtime not started: acreate_client missing (supabase_py_version={SUPABASE_PY_VERSION}). "
            "Please upgrade 'supabase' package to >=2.6.0."
        )
        return

    # mark (re)start and reset flags
    st.session_state["_rt_started"] = True
    _RT_FLAGS.update({
        "client_created": False,
        "channel_subscribed": False,
        "connect_ok": False,
        "last_error": None,
        "phase": "starting",
    })

    st.session_state.setdefault("_rt_debug", {})
    st.session_state["_rt_debug"].update({
        "acreate_client_available": acreate_client is not None,
        "async_client_available": AsyncClient is not None,
        "started": True,
        "client_created": False,
        "channel_subscribed": False,
        "connect_ok": False,
        "last_event_ts": None,
        "last_event_human": None,
        "last_error": None,
        "supabase_py_version": SUPABASE_PY_VERSION,
        "phase": "starting",
    })

    def _worker():
        try:
            _RT_FLAGS["phase"] = "worker_start"

            async def run():
                try:
                    # Create async client (required for Realtime in Python)
                    _RT_FLAGS["phase"] = "create_client"
                    acli = await acreate_client(SUPABASE_URL, SUPABASE_KEY)
                    _RT_FLAGS["client_created"] = True

                    # Optional quick probe (non-fatal)
                    try:
                        _RT_FLAGS["phase"] = "probe_rest"
                        _ = await acli.table("players").select("name").limit(1).execute()
                    except Exception as e_probe:
                        _RT_FLAGS["last_error"] = f"probe_rest: {type(e_probe).__name__}: {e_probe}"

                    # Callback writes into queue (no session_state from thread)
                    def on_change(payload):
                        try:
                            _RT_QUEUE.put(time.time())
                        except Exception:
                            pass

                    # Connect → subscribe → listen (per docs)
                    _RT_FLAGS["phase"] = "connect"
                    await acli.realtime.connect()
                    _RT_FLAGS["connect_ok"] = True

                    _RT_FLAGS["phase"] = "subscribe"
                    chan = acli.channel("tt_elo_changes")
                    chan.on_postgres_changes("*", schema="public", table="pending_matches", callback=on_change)
                    chan.on_postgres_changes("*", schema="public", table="pending_doubles", callback=on_change)
                    chan.on_postgres_changes("*", schema="public", table="pending_rounds", callback=on_change)
                    chan.on_postgres_changes("*", schema="public", table="matches", callback=on_change)
                    chan.on_postgres_changes("*", schema="public", table="doubles", callback=on_change)
                    chan.on_postgres_changes("*", schema="public", table="rounds", callback=on_change)
                    await chan.subscribe()
                    _RT_FLAGS["channel_subscribed"] = True

                    _RT_FLAGS["phase"] = "listen"
                    await acli.realtime.listen()
                except Exception as e:
                    _RT_FLAGS["last_error"] = f"error at phase={_RT_FLAGS.get('phase')}: {type(e).__name__}: {e}\n{traceback.format_exc()}"
                    st.session_state["_rt_started"] = False

            # Run the async routine in this background thread
            asyncio.run(run())
        except Exception as e_outer:
            _RT_FLAGS["last_error"] = f"worker_error: {type(e_outer).__name__}: {e_outer}\n{traceback.format_exc()}"
            st.session_state["_rt_started"] = False

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    st.session_state["_rt_thread"] = t

# Kick off realtime subscriber once per session
_ensure_realtime_started()

# Mirror thread flags into debug (main thread only)
dbg = st.session_state.get("_rt_debug", {})
dbg["started"] = True if st.session_state.get("_rt_started") else False
dbg["client_created"] = bool(_RT_FLAGS.get("client_created"))
dbg["channel_subscribed"] = bool(_RT_FLAGS.get("channel_subscribed"))
dbg["connect_ok"] = bool(_RT_FLAGS.get("connect_ok"))
dbg["last_error"] = _RT_FLAGS.get("last_error")
dbg["phase"] = _RT_FLAGS.get("phase", dbg.get("phase"))
st.session_state["_rt_debug"] = dbg

# Drain realtime event queue and rerun once per batch
_got_event = False
while True:
    try:
        _ = _RT_QUEUE.get_nowait()
        _got_event = True
    except Exception:
        break

if _got_event:
    try:
        st.cache_data.clear()
    except Exception:
        pass
    st.session_state["_rt_debug"]["last_event_human"] = datetime.now(BERLIN).strftime("%d.%m.%Y %H:%M:%S")
    st.rerun()
# endregion
# endregion

# region ELO Core (Math + Apply)
# --- Core math identical to old system ---
def _expected(r_a: float, r_b: float) -> float:
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))

def _calc_elo(r_a: float, r_b: float, score_a: float, k: float) -> int:
    return int(round(r_a + k * (score_a - _expected(r_a, r_b))))

def _compute_gelo(single: float, d_elo: float, r_elo: float,
                  w_e: float = 0.6, w_d: float = 0.25, w_r: float = 0.15) -> int:
    return int(round(w_e * single + w_d * d_elo + w_r * r_elo))

# --- Weighted by games per mode ---
def _compute_gelo_weighted_by_games(row: dict, new_single=None, new_d=None, new_r=None,
                                    inc_single: int = 0, inc_double: int = 0, inc_round: int = 0) -> int:
    """Compute overall g_elo weighted strictly by number of games per mode.
    Weights: wE = (spiele+inc_single) / N, wD = (d_spiele+inc_double) / N, wR = (r_spiele+inc_round) / N.
    If N == 0 -> 1200.
    new_single/new_d/new_r allow passing freshly updated ratings for the current event.
    """
    rE = float(new_single if new_single is not None else row.get("elo", 1200))
    rD = float(new_d      if new_d      is not None else row.get("d_elo", 1200))
    rR = float(new_r      if new_r      is not None else row.get("r_elo", 1200))

    nE = int(row.get("spiele", 0))   + int(inc_single)
    nD = int(row.get("d_spiele", 0)) + int(inc_double)
    nR = int(row.get("r_spiele", 0)) + int(inc_round)
    N = nE + nD + nR
    if N <= 0:
        return 1200

    g = (nE * rE + nD * rD + nR * rR) / N
    return int(round(g))

def _get_player_row(name: str) -> dict:
    res = supabase.table("players").select("*").eq("name", name).single().execute()
    return res.data or {}

def _update_player_fields(name: str, fields: dict):
    supabase.table("players").update(fields).eq("name", name).execute()

 # --- Incremental apply: Einzel ---
def apply_single_result(a: str, b: str, punktea: int, punkteb: int,
                        k_base: int = 64, datum: str | None = None):
    if a == b:
        return False, "Spieler A und B müssen unterschiedlich sein."
    if punktea == punkteb:
        return False, "Unentschieden ist nicht erlaubt."
    if min(punktea, punkteb) < 0:
        return False, "Punktzahlen müssen >= 0 sein."

    row_a = _get_player_row(a)
    row_b = _get_player_row(b)
    if not row_a or not row_b:
        return False, "Spieler nicht gefunden."

    r_a = float(row_a.get("elo", 1200))
    r_b = float(row_b.get("elo", 1200))
    d_a = float(row_a.get("d_elo", 1200))
    d_b = float(row_b.get("d_elo", 1200))
    rra = float(row_a.get("r_elo", 1200))
    rrb = float(row_b.get("r_elo", 1200))

    margin = abs(int(punktea) - int(punkteb))
    k_eff = k_base * (1 + margin / 11)
    a_wins = punktea > punkteb

    new_r_a = _calc_elo(r_a, r_b, 1 if a_wins else 0, k_eff)
    new_r_b = _calc_elo(r_b, r_a, 0 if a_wins else 1, k_eff)

    # Stats-Updates
    a_siege = int(row_a.get("siege", 0)) + (1 if a_wins else 0)
    a_nied  = int(row_a.get("niederlagen", 0)) + (0 if a_wins else 1)
    a_games = int(row_a.get("spiele", 0)) + 1
    b_siege = int(row_b.get("siege", 0)) + (0 if a_wins else 1)
    b_nied  = int(row_b.get("niederlagen", 0)) + (1 if a_wins else 0)
    b_games = int(row_b.get("spiele", 0)) + 1

    new_g_a = _compute_gelo_weighted_by_games(row_a, new_single=new_r_a, inc_single=1)
    new_g_b = _compute_gelo_weighted_by_games(row_b, new_single=new_r_b, inc_single=1)

    _update_player_fields(a, {
        "elo": new_r_a, "siege": a_siege, "niederlagen": a_nied, "spiele": a_games, "g_elo": new_g_a
    })
    _update_player_fields(b, {
        "elo": new_r_b, "siege": b_siege, "niederlagen": b_nied, "spiele": b_games, "g_elo": new_g_b
    })

    # Match speichern
    supabase.table("matches").insert({
        "datum": datum or now_utc_iso(),
        "a": a, "b": b, "punktea": int(punktea), "punkteb": int(punkteb)
    }).execute()

    return True, "Einzelmatch gespeichert und ELO aktualisiert."

# --- Incremental apply: Doppel ---
def apply_double_result(a1: str, a2: str, b1: str, b2: str, punktea: int, punkteb: int,
                        k_base: int = 48):
    for n in (a1, a2, b1, b2):
        if not _get_player_row(n):
            return False, f"Spieler {n} nicht gefunden."
    if punktea == punkteb:
        return False, "Unentschieden ist nicht erlaubt."

    ra1 = float(_get_player_row(a1).get("d_elo", 1200))
    ra2 = float(_get_player_row(a2).get("d_elo", 1200))
    rb1 = float(_get_player_row(b1).get("d_elo", 1200))
    rb2 = float(_get_player_row(b2).get("d_elo", 1200))
    a_avg, b_avg = (ra1 + ra2) / 2, (rb1 + rb2) / 2

    margin = abs(int(punktea) - int(punkteb))
    k_eff = k_base * (1 + margin / 11)
    team_a_win = 1 if punktea > punkteb else 0

    def _team_update(name: str, old_d: float, opp_avg: float, s: int):
        exp = 1 / (1 + 10 ** ((opp_avg - old_d) / 400))
        delta = k_eff * (s - exp)
        return int(round(old_d + delta))

    nr_a1 = _team_update(a1, ra1, b_avg, team_a_win)
    nr_a2 = _team_update(a2, ra2, b_avg, team_a_win)
    nr_b1 = _team_update(b1, rb1, a_avg, 1 - team_a_win)
    nr_b2 = _team_update(b2, rb2, a_avg, 1 - team_a_win)

    def _apply_one(name: str, new_d: int, s: int):
        row = _get_player_row(name)
        g = _compute_gelo_weighted_by_games(row, new_d=new_d, inc_double=1)
        _update_player_fields(name, {
            "d_elo": new_d,
            "d_siege": int(row.get("d_siege", 0)) + s,
            "d_niederlagen": int(row.get("d_niederlagen", 0)) + (1 - s),
            "d_spiele": int(row.get("d_spiele", 0)) + 1,
            "g_elo": g,
        })

    _apply_one(a1, nr_a1, team_a_win)
    _apply_one(a2, nr_a2, team_a_win)
    _apply_one(b1, nr_b1, 1 - team_a_win)
    _apply_one(b2, nr_b2, 1 - team_a_win)

    supabase.table("doubles").insert({
        "datum": now_utc_iso(),
        "a1": a1, "a2": a2, "b1": b1, "b2": b2,
        "punktea": int(punktea), "punkteb": int(punkteb)
    }).execute()

    return True, "Doppel gespeichert und ELO aktualisiert."

# --- Incremental apply: Rundlauf ---
def apply_round_result(teilnehmer: list, finalisten: tuple, sieger: str, k_base: int = 48):
    if not teilnehmer or sieger not in teilnehmer:
        return False, "Teilnehmer/Sieger ungültig."
    f1, f2 = (finalisten + (None, None))[:2] if isinstance(finalisten, tuple) else (None, None)
    # Require two distinct finalists that are part of the participants,
    # and ensure the winner is one of the finalists.
    if not f1 or not f2:
        return False, "Bitte beide Finalisten angeben."
    if f1 == f2:
        return False, "Finalisten müssen unterschiedlich sein."
    if f1 not in teilnehmer or f2 not in teilnehmer:
        return False, "Finalisten müssen Teilnehmer sein."
    if sieger not in (f1, f2):
        return False, "Sieger muss einer der Finalisten sein."
    # Durchschnitts-Rating zum Zeitpunkt
    r_values = []
    for p in teilnehmer:
        row = _get_player_row(p)
        if not row:
            return False, f"Spieler {p} nicht gefunden."
        r_values.append(float(row.get("r_elo", 1200)))
    avg = sum(r_values) / len(r_values)

    deltas = {}
    for p in teilnehmer:
        row = _get_player_row(p)
        old = float(row.get("r_elo", 1200))
        if p == sieger:
            s = 1
            r_siege = int(row.get("r_siege", 0)) + 1
            _update_player_fields(p, {"r_siege": r_siege})
        elif p in (f1, f2):
            s = 0.5
            r_zweite = int(row.get("r_zweite", 0)) + 1
            _update_player_fields(p, {"r_zweite": r_zweite})
        else:
            s = 0
            r_n = int(row.get("r_niederlagen", 0)) + 1
            _update_player_fields(p, {"r_niederlagen": r_n})
        exp = _expected(old, avg)
        deltas[p] = k_base * (s - exp)

    offset = sum(deltas.values()) / len(deltas) if deltas else 0
    for p, delta in deltas.items():
        row = _get_player_row(p)
        new_r = int(round(float(row.get("r_elo", 1200)) + (delta - offset)))
        g = _compute_gelo_weighted_by_games(row, new_r=new_r, inc_round=1)
        _update_player_fields(p, {
            "r_elo": new_r,
            "r_spiele": int(row.get("r_spiele", 0)) + 1,
            "g_elo": g,
        })

    supabase.table("rounds").insert({
        "datum": now_utc_iso(),
        "teilnehmer": ";".join(teilnehmer),
        "finalisten": ";".join([f for f in (f1, f2) if f]),
        "sieger": sieger,
    }).execute()

    return True, "Rundlauf gespeichert und ELO aktualisiert."
# endregion

# region Pending helpers (Einzel / Doppel / Rundlauf)

@st.cache_data(ttl=5)
def load_last_matches(limit: int = 5):
    return supabase.table("matches").select("*").order("datum", desc=True).limit(limit).execute().data or []


def submit_single_pending(creator: str, a: str, b: str, punktea: int, punkteb: int):
    if a == b:
        return False, "Spieler A und B müssen unterschiedlich sein."
    if min(punktea, punkteb) < 0:
        return False, "Punktzahlen müssen >= 0 sein."
    # determine which side the creator is on
    confa = creator == a
    confb = creator == b
    if not (confa or confb):
        # if creator is neither A nor B, auto-confirm A side
        confa = True
    supabase.table("pending_matches").insert({
        "datum": now_utc_iso(),
        "a": a, "b": b,
        "punktea": int(punktea), "punkteb": int(punkteb),
        "confa": confa, "confb": confb,
    }).execute()
    return True, "Einzelmatch eingereicht. Warte auf Bestätigung."


@st.cache_data(ttl=5)
def fetch_pending_for_user(user: str):
    # Einzel: pending matches where user participates
    res = supabase.table("pending_matches").select("*").order("datum", desc=True).execute()
    rows = res.data or []
    to_confirm, waiting = [], []
    for r in rows:
        a, b = r.get("a"), r.get("b")
        if user not in (a, b):
            continue
        confa = bool(r.get("confa", False))
        confb = bool(r.get("confb", False))
        if a == user and not confa:
            to_confirm.append(r)
        elif b == user and not confb:
            to_confirm.append(r)
        elif a == user and confa and not confb:
            waiting.append(r)
        elif b == user and confb and not confa:
            waiting.append(r)
    return to_confirm, waiting


def confirm_pending_match(row_id: str, user: str):
    # fetch row
    r = supabase.table("pending_matches").select("*").eq("id", row_id).single().execute().data
    if not r:
        return False, "Eintrag nicht gefunden."
    fields = {}
    if r["a"] == user:
        fields["confa"] = True
    if r["b"] == user:
        fields["confb"] = True
    if not fields:
        return False, "Du bist an diesem Match nicht beteiligt."
    supabase.table("pending_matches").update(fields).eq("id", row_id).execute()

    # Reload row to check completion
    r = supabase.table("pending_matches").select("*").eq("id", row_id).single().execute().data
    if r and r.get("confa") and r.get("confb"):
        # finalize: apply elo & move to matches
        ok, msg = apply_single_result(r["a"], r["b"], int(r["punktea"]), int(r["punkteb"]), datum=r.get("datum"))
        # delete pending
        supabase.table("pending_matches").delete().eq("id", row_id).execute()
        try:
            st.cache_data.clear()
        except Exception:
            pass
        return ok, "Match bestätigt und gewertet."
    try:
        st.cache_data.clear()
    except Exception:
        pass
    return True, "Bestätigung gespeichert. Warte auf Gegner."

# New function: reject_pending_match
def reject_pending_match(row_id: str, user: str):
    """Allow either participant (creator or opponent) to reject a pending match.
    Deletes the pending entry and does not change ratings or create a match record.
    """
    r = supabase.table("pending_matches").select("*").eq("id", row_id).single().execute().data
    if not r:
        return False, "Eintrag nicht gefunden."
    if r.get("a") != user and r.get("b") != user:
        return False, "Du bist an diesem Match nicht beteiligt."
    supabase.table("pending_matches").delete().eq("id", row_id).execute()
    # Invalidate cached pending/matches lists
    try:
        st.cache_data.clear()
    except Exception:
        pass
    return True, "Match abgelehnt und entfernt."
# endregion

# region Pending helpers (Doppel)
@st.cache_data(ttl=5)
def fetch_pending_doubles_for_user(user: str):
    res = supabase.table("pending_doubles").select("*").order("datum", desc=True).execute()
    rows = res.data or []
    to_confirm, waiting = [], []
    for r in rows:
        a_side = user in (r.get("a1"), r.get("a2"))
        b_side = user in (r.get("b1"), r.get("b2"))
        if not (a_side or b_side):
            continue
        if a_side and not r.get("confa", False):
            to_confirm.append(r)
        elif b_side and not r.get("confb", False):
            to_confirm.append(r)
        elif (a_side and r.get("confa", False) and not r.get("confb", False)) or \
             (b_side and r.get("confb", False) and not r.get("confa", False)):
            waiting.append(r)
    return to_confirm, waiting


def submit_double_pending(creator: str, a1: str, a2: str, b1: str, b2: str, pa: int, pb: int):
    names = [a1, a2, b1, b2]
    if any(n is None or n == "" for n in names):
        return False, "Alle Spieler müssen ausgewählt sein."
    if len(set(names)) < 4:
        return False, "Jeder Spieler darf nur einmal vorkommen."
    if pa == pb:
        return False, "Unentschieden ist nicht erlaubt."
    # confirmation side
    confa = creator in (a1, a2)
    confb = creator in (b1, b2)
    if not (confa or confb):
        confa = True
    supabase.table("pending_doubles").insert({
        "datum": now_utc_iso(),
        "a1": a1, "a2": a2, "b1": b1, "b2": b2,
        "punktea": int(pa), "punkteb": int(pb),
        "confa": confa, "confb": confb,
    }).execute()
    try:
        st.cache_data.clear()
    except Exception:
        pass
    return True, "Doppel eingereicht. Warte auf Bestätigung."


def confirm_pending_double(row_id: str, user: str):
    r = supabase.table("pending_doubles").select("*").eq("id", row_id).single().execute().data
    if not r:
        return False, "Eintrag nicht gefunden."
    fields = {}
    if user in (r.get("a1"), r.get("a2")):
        fields["confa"] = True
    if user in (r.get("b1"), r.get("b2")):
        fields["confb"] = True
    if not fields:
        return False, "Du bist an diesem Match nicht beteiligt."
    supabase.table("pending_doubles").update(fields).eq("id", row_id).execute()

    r = supabase.table("pending_doubles").select("*").eq("id", row_id).single().execute().data
    if r and r.get("confa") and r.get("confb"):
        ok, msg = apply_double_result(r["a1"], r["a2"], r["b1"], r["b2"], int(r["punktea"]), int(r["punkteb"]))
        supabase.table("pending_doubles").delete().eq("id", row_id).execute()
        try:
            st.cache_data.clear()
        except Exception:
            pass
        return ok, "Doppel bestätigt und gewertet."
    try:
        st.cache_data.clear()
    except Exception:
        pass
    return True, "Bestätigung gespeichert. Warte auf Gegner."


def reject_pending_double(row_id: str, user: str):
    r = supabase.table("pending_doubles").select("*").eq("id", row_id).single().execute().data
    if not r:
        return False, "Eintrag nicht gefunden."
    if user not in (r.get("a1"), r.get("a2"), r.get("b1"), r.get("b2")):
        return False, "Du bist an diesem Match nicht beteiligt."
    supabase.table("pending_doubles").delete().eq("id", row_id).execute()
    try:
        st.cache_data.clear()
    except Exception:
        pass
    return True, "Doppel abgelehnt und entfernt."
# endregion

# region Pending helpers (Rundlauf)
@st.cache_data(ttl=5)
def fetch_pending_rounds_for_user(user: str):
    res = supabase.table("pending_rounds").select("*").order("datum", desc=True).execute()
    rows = res.data or []
    to_confirm, waiting = [], []
    for r in rows:
        teilnehmer = str(r.get("teilnehmer", "")).split(";") if r.get("teilnehmer") else []
        if user not in teilnehmer:
            continue
        creator = r.get("creator")
        confa = bool(r.get("confa", False))
        confb = bool(r.get("confb", False))
        if creator == user:
            # Ersteller: nur warten, wenn die eigene (erste) Bestätigung erfolgt ist und die zweite fehlt
            if confa and not confb:
                waiting.append(r)
        else:
            # Andere Teilnehmer: bestätigen solange nicht vollständig bestätigt
            if not (confa and confb):
                to_confirm.append(r)
    return to_confirm, waiting


def submit_round_pending(creator: str, teilnehmer: list[str], finalisten: tuple[str|None,str|None], sieger: str):
    teilnehmer = [t for t in (teilnehmer or []) if t]
    if len(teilnehmer) < 3:
        return False, "Mindestens 3 Teilnehmer erforderlich."
    if not sieger or sieger not in teilnehmer:
        return False, "Sieger muss Teilnehmer sein."
    f1, f2 = finalisten if isinstance(finalisten, tuple) else (None, None)
    if not f1 or not f2:
        return False, "Bitte beide Finalisten wählen."
    if f1 == f2:
        return False, "Finalisten müssen unterschiedlich sein."
    if f1 not in teilnehmer or f2 not in teilnehmer:
        return False, "Finalisten müssen Teilnehmer sein."
    if sieger not in (f1, f2):
        return False, "Sieger muss einer der Finalisten sein."
    confa = creator in teilnehmer
    confb = False
    supabase.table("pending_rounds").insert({
        "datum": now_utc_iso(),
        "teilnehmer": ";".join(teilnehmer),
        "finalisten": ";".join([f1, f2]),
        "sieger": sieger,
        "creator": creator,
        "confa": confa, "confb": confb,
    }).execute()
    try:
        st.cache_data.clear()
    except Exception:
        pass
    return True, "Rundlauf eingereicht. Warte auf Bestätigung."


def confirm_pending_round(row_id: str, user: str):
    r = supabase.table("pending_rounds").select("*").eq("id", row_id).single().execute().data
    if r.get("creator") == user and r.get("confa") and not r.get("confb"):
        return False, "Bestätigung muss durch einen anderen Teilnehmer erfolgen."
    if not r:
        return False, "Eintrag nicht gefunden."
    teilnehmer = str(r.get("teilnehmer", "")).split(";") if r.get("teilnehmer") else []
    if user not in teilnehmer:
        return False, "Du bist an diesem Rundlauf nicht beteiligt."
    fields = {}
    if not r.get("confa", False):
        fields["confa"] = True
    elif not r.get("confb", False):
        fields["confb"] = True
    else:
        return True, "Schon vollständig bestätigt."
    supabase.table("pending_rounds").update(fields).eq("id", row_id).execute()

    r = supabase.table("pending_rounds").select("*").eq("id", row_id).single().execute().data
    if r and r.get("confa") and r.get("confb"):
        teilnehmer = str(r.get("teilnehmer", "")).split(";") if r.get("teilnehmer") else []
        fin = str(r.get("finalisten", "")).split(";") if r.get("finalisten") else []
        f1, f2 = (fin + [None, None])[:2]
        ok, msg = apply_round_result(teilnehmer, (f1, f2), r.get("sieger"))
        supabase.table("pending_rounds").delete().eq("id", row_id).execute()
        try:
            st.cache_data.clear()
        except Exception:
            pass
        return ok, "Rundlauf bestätigt und gewertet."
    try:
        st.cache_data.clear()
    except Exception:
        pass
    return True, "Bestätigung gespeichert. Warte auf Gegner."


def reject_pending_round(row_id: str, user: str):
    r = supabase.table("pending_rounds").select("*").eq("id", row_id).single().execute().data
    if not r:
        return False, "Eintrag nicht gefunden."
    teilnehmer = str(r.get("teilnehmer", "")).split(";") if r.get("teilnehmer") else []
    if user not in teilnehmer:
        return False, "Du bist an diesem Rundlauf nicht beteiligt."
    supabase.table("pending_rounds").delete().eq("id", row_id).execute()
    try:
        st.cache_data.clear()
    except Exception:
        pass
    return True, "Rundlauf abgelehnt und entfernt."
# endregion

# region Pending helpers (Combined View)
@st.cache_data(ttl=5)
def pending_combined_for_user(user: str):
    e_to, e_wait = fetch_pending_for_user(user)
    d_to, d_wait = fetch_pending_doubles_for_user(user)
    r_to, r_wait = fetch_pending_rounds_for_user(user)

    def map_single(rows):
        return [{
            "mode": "Einzel",
            "id": r["id"],
            "datum": r.get("datum"),
            "text": f"{r['a']} vs {r['b']} — {r['punktea']}:{r['punkteb']}"
        } for r in rows]

    def map_double(rows):
        return [{
            "mode": "Doppel",
            "id": r["id"],
            "datum": r.get("datum"),
            "text": f"A: {r['a1']} & {r['a2']} — B: {r['b1']} & {r['b2']} — {r['punktea']}:{r['punkteb']}"
        } for r in rows]

    def map_round(rows):
        return [{
            "mode": "Rundlauf",
            "id": r["id"],
            "datum": r.get("datum"),
            "text": f"Teilnehmer: {r['teilnehmer']} — Sieger: {r['sieger']}"
        } for r in rows]

    to_confirm = map_single(e_to) + map_double(d_to) + map_round(r_to)
    waiting    = map_single(e_wait) + map_double(d_wait) + map_round(r_wait)

    # Sort by datum desc if available
    def _sort_key(x):
        return x.get("datum") or ""
    to_confirm = sorted(to_confirm, key=_sort_key, reverse=True)
    waiting    = sorted(waiting, key=_sort_key, reverse=True)
    return to_confirm, waiting
# endregion

# region Persistent Login via Query Params
q = st.query_params
def _qp_first(val):
    if isinstance(val, list):
        return val[0] if val else None
    return val
if "user" not in st.session_state:
    auto_user = _qp_first(q.get("user"))
    auto_token = _qp_first(q.get("token"))
    if auto_user and auto_token:
        # Fetch stored hash for auto_user
        resp = supabase.table("players").select("pin").eq("name", auto_user).single().execute()
        stored_hash = resp.data.get("pin") if resp.data else None
        if stored_hash and auto_token == stored_hash:
            st.session_state.user = auto_user
# endregion

# region Authentication & CSV Display
if 'user' not in st.session_state:
    st.header("🔐 Login oder Registrierung")
    login_tab, register_tab = st.tabs(["Login", "Registrieren"])

    with login_tab:
        with st.form("login_form"):
            name = st.text_input("Name")
            pin  = st.text_input("PIN", type="password")
            remember = st.checkbox("Angemeldet bleiben", value=True)
            if st.form_submit_button("Login"):
                resp = supabase.table("players").select("pin").eq("name", name).single().execute()
                stored_hash = resp.data.get("pin") if resp.data else None
                if stored_hash and check_pin(pin, stored_hash):
                    # Successful login
                    st.session_state.user = name
                    if remember:
                        st.query_params["user"] = name
                        st.query_params["token"] = stored_hash
                    else:
                        # Do not persist in URL
                        if "user" in st.query_params:
                            del st.query_params["user"]
                        if "token" in st.query_params:
                            del st.query_params["token"]
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
                    hashed = hash_pin(pin)
                    supabase.table("players").insert({"name": name, "pin": hashed}).execute()
                    st.success("Registrierung erfolgreich. Bitte einloggen.")
    st.stop()
# region UI: Tabs (Willkommen, Spielen, Account)
else:
    st.header("🏓 AK-Tischtennis")
    
    st.session_state.setdefault("editing", False)

    main_tab1, main_tab2, main_tab3 = st.tabs(["Übersicht", "Spielen", "Account"])

    with main_tab1:
        # Willkommen: ELO Anzeige, Matches bestätigen, letzte 5 Spiele
        st.subheader(f"👋 Willkommen, {st.session_state.user}!")
        st.subheader("Deine Ratings")
        me = supabase.table("players").select("*").eq("name", st.session_state.user).single().execute().data
        if me:
            cols = st.columns(4)
            cols[0].metric("Gesamt-ELO", int(me.get("g_elo", 1200)))
            cols[1].metric("Einzel", int(me.get("elo", 1200)))
            cols[2].metric("Doppel", int(me.get("d_elo", 1200)))
            cols[3].metric("Rundlauf", int(me.get("r_elo", 1200)))
        st.divider()

        st.subheader("✅ Offene Bestätigungen")

        to_conf_all, wait_all = pending_combined_for_user(st.session_state.user)
        if to_conf_all:
            for it in to_conf_all:
                c1, c2, c3, c4 = st.columns([5,2,2,2])
                c1.write(f"[{it['mode']}] {it['text']}")
                c2.write(fmt_dt_local(it['datum']))
                if c3.button("✅", key=f"w_conf_{it['mode']}_{it['id']}"):
                    if it['mode'] == 'Einzel':
                        ok, msg = confirm_pending_match(it['id'], st.session_state.user)
                    elif it['mode'] == 'Doppel':
                        ok, msg = confirm_pending_double(it['id'], st.session_state.user)
                    else:
                        ok, msg = confirm_pending_round(it['id'], st.session_state.user)
                    if ok:
                        st.success(f"{it['mode']} bestätigt")
                    else:
                        st.error(msg)
                    st.rerun()
                if c4.button("❌", key=f"w_rej_{it['mode']}_{it['id']}"):
                    if it['mode'] == 'Einzel':
                        ok, msg = reject_pending_match(it['id'], st.session_state.user)
                    elif it['mode'] == 'Doppel':
                        ok, msg = reject_pending_double(it['id'], st.session_state.user)
                    else:
                        ok, msg = reject_pending_round(it['id'], st.session_state.user)
                    if ok:
                        st.success(f"{it['mode']} abgelehnt")
                    else:
                        st.error(msg)
                    st.rerun()
        else:
            st.caption("Nichts zu bestätigen.")

        st.subheader("Letzte Spiele")

        df_last = load_last_events_table(5)
        if not df_last.empty:
            # allow multiline cells (render \n as line breaks) and hide index
            st.markdown(
                "<style>[data-testid='stDataFrame'] div[role='gridcell']{white-space:pre-wrap;}</style>",
                unsafe_allow_html=True
            )
            st.dataframe(df_last, use_container_width=True, hide_index=True)
        else:
            st.info("Noch keine Spiele vorhanden.")

    with main_tab2:
        # Spielen: Match eintragen (Subtabs) + Bestätigen + Ausstehend
        sub1, sub2, sub3 = st.tabs(["Einzel", "Doppel", "Rundlauf"])

        with sub1:
            st.markdown("### Einzel eintragen")
            data_players = supabase.table("players").select("name").order("name").execute().data
            names = [p["name"] for p in data_players] if data_players else []
            if len(names) >= 2:
                with st.form("single_form", clear_on_submit=False):
                    c1, c2 = st.columns(2)
                    with c1:
                        a = st.selectbox("Spieler A", names, key="ein_a")
                        pa = st.number_input("Punkte A", min_value=0, step=1, key="ein_pa")
                    with c2:
                        b = st.selectbox("Spieler B", names, index=1 if len(names)>1 else 0, key="ein_b")
                        pb = st.number_input("Punkte B", min_value=0, step=1, key="ein_pb")
                    submit_single = st.form_submit_button("✅ Bestätigen")
                    if submit_single:
                        if not a or not b:
                            st.error("Bitte beide Spieler auswählen.")
                        elif a == b:
                            st.error("Spieler A und B müssen unterschiedlich sein.")
                        else:
                            ok, msg = submit_single_pending(st.session_state.user, a, b, int(pa), int(pb))
                            if ok:
                                st.success("Match angelegt")
                            else:
                                st.error(msg)
                            st.rerun()
            else:
                st.info("Mindestens zwei Spieler erforderlich.")

        with sub2:
            st.markdown("### Doppel eintragen")
            data_players = supabase.table("players").select("name").order("name").execute().data
            names = [p["name"] for p in data_players] if data_players else []
            if len(names) >= 4:
                with st.form("double_form", clear_on_submit=False):
                    c1, c2 = st.columns(2)
                    with c1:
                        a1 = st.selectbox("Team A – Spieler 1", names, key="d_a1")
                        opts_a2 = [n for n in names if n != a1]
                        a2 = st.selectbox("Team A – Spieler 2", opts_a2, key="d_a2")
                        pa = st.number_input("Punkte Team A", min_value=0, step=1, key="d_pa")
                    with c2:
                        opts_b1 = [n for n in names if n not in {a1, a2}]
                        b1 = st.selectbox("Team B – Spieler 1", opts_b1, key="d_b1")
                        opts_b2 = [n for n in names if n not in {a1, a2, b1}]
                        b2 = st.selectbox("Team B – Spieler 2", opts_b2, key="d_b2")
                        pb = st.number_input("Punkte Team B", min_value=0, step=1, key="d_pb")

                    disable_submit = not all([a1, a2, b1, b2]) or len({a1, a2, b1, b2}) < 4
                    submit_double = st.form_submit_button("✅ Doppel einreichen", disabled=disable_submit)
                    if submit_double:
                        ok, msg = submit_double_pending(st.session_state.user, a1, a2, b1, b2, int(pa), int(pb))
                        if ok:
                            st.success("Doppel angelegt")
                        else:
                            st.error(msg)
                        st.rerun()
            else:
                st.info("Mindestens vier Spieler erforderlich.")

        with sub3:
            st.markdown("### Rundlauf eintragen")
            data_players = supabase.table("players").select("name").order("name").execute().data
            names = [p["name"] for p in data_players] if data_players else []
            if len(names) >= 3:
                with st.form("round_form", clear_on_submit=False):
                    participants = st.multiselect("Teilnehmer", names, key="r_parts")
                    fin_cols = st.columns(2)
                    with fin_cols[0]:
                        fin1 = st.selectbox("Finalist 1", [""] + participants, key="r_f1")
                    with fin_cols[1]:
                        fin2 = st.selectbox("Finalist 2", [""] + participants, key="r_f2")
                    winner_options = [x for x in [fin1, fin2] if x]
                    winner = st.selectbox("Sieger", winner_options if winner_options else [""], key="r_win")

                    submit_round = st.form_submit_button("✅ Rundlauf einreichen")
                    if submit_round:
                        if len(participants) < 3:
                            st.error("Mindestens drei Teilnehmer erforderlich.")
                        elif not winner or winner not in participants:
                            st.error("Sieger muss Teilnehmer sein.")
                        elif not fin1 or not fin2:
                            st.error("Bitte beide Finalisten wählen.")
                        elif fin1 == fin2:
                            st.error("Finalisten müssen unterschiedlich sein.")
                        elif fin1 not in participants or fin2 not in participants:
                            st.error("Finalisten müssen Teilnehmer sein.")
                        elif winner not in (fin1, fin2):
                            st.error("Sieger muss einer der Finalisten sein.")
                        else:
                            ok, msg = submit_round_pending(st.session_state.user, participants, (fin1, fin2), winner)
                            if ok:
                                st.success("Rundlauf angelegt")
                            else:
                                st.error(msg)
                            st.rerun()
            else:
                st.info("Mindestens drei Spieler erforderlich.")

        col_ref3, _ = st.columns([1, 9])

        st.markdown("### ✅ Offene Bestätigungen")
        to_conf_all, wait_all = pending_combined_for_user(st.session_state.user)
        if to_conf_all:
            for it in to_conf_all:
                c1, c2, c3, c4 = st.columns([5,2,2,2])
                c1.write(f"[{it['mode']}] {it['text']}")
                c2.write(fmt_dt_local(it['datum']))
                if c3.button("✅", key=f"s_conf_{it['mode']}_{it['id']}"):
                    if it['mode'] == 'Einzel':
                        ok, msg = confirm_pending_match(it['id'], st.session_state.user)
                    elif it['mode'] == 'Doppel':
                        ok, msg = confirm_pending_double(it['id'], st.session_state.user)
                    else:
                        ok, msg = confirm_pending_round(it['id'], st.session_state.user)
                    if ok:
                        st.success(f"{it['mode']} bestätigt")
                    else:
                        st.error(msg)
                    st.rerun()
                if c4.button("❌", key=f"s_rej_{it['mode']}_{it['id']}"):
                    if it['mode'] == 'Einzel':
                        ok, msg = reject_pending_match(it['id'], st.session_state.user)
                    elif it['mode'] == 'Doppel':
                        ok, msg = reject_pending_double(it['id'], st.session_state.user)
                    else:
                        ok, msg = reject_pending_round(it['id'], st.session_state.user)
                    if ok:
                        st.success(f"{it['mode']} abgelehnt")
                    else:
                        st.error(msg)
                    st.rerun()
        else:
            st.caption("Nichts zu bestätigen.")

        st.markdown("### 🕔 Vom Gegner ausstehend")
        if wait_all:
            for it in wait_all:
                c1, c2, c3 = st.columns([5,2,2])
                c1.write(f"[{it['mode']}] {it['text']}")
                c2.write(fmt_dt_local(it['datum']))
                if c3.button("❌", key=f"s_rej_wait_{it['mode']}_{it['id']}"):
                    if it['mode'] == 'Einzel':
                        ok, msg = reject_pending_match(it['id'], st.session_state.user)
                    elif it['mode'] == 'Doppel':
                        ok, msg = reject_pending_double(it['id'], st.session_state.user)
                    else:
                        ok, msg = reject_pending_round(it['id'], st.session_state.user)
                    if ok:
                        st.success(f"{it['mode']} abgelehnt")
                    else:
                        st.error(msg)
                    st.rerun()
        else:
            st.caption("Keine offenen Anfragen beim Gegner.")

    with main_tab3:
        with st.expander("⚡ Realtime-Debug", expanded=False):
            st.caption(f"supabase_py_version(var) = {SUPABASE_PY_VERSION}; acreate_client is None = {acreate_client is None}")
            dbg = st.session_state.get("_rt_debug", {})
            _t = st.session_state.get("_rt_thread")
            _alive = bool(_t.is_alive()) if _t is not None else False
            st.write({
                "supabase_py_version": dbg.get("supabase_py_version"),
                "acreate_client_available": dbg.get("acreate_client_available"),
                "acreate_is_coro": dbg.get("acreate_is_coro"),
                "async_client_available": dbg.get("async_client_available"),
                "started": dbg.get("started"),
                "client_created": dbg.get("client_created"),
                "channel_subscribed": dbg.get("channel_subscribed"),
                "connect_ok": dbg.get("connect_ok"),
                "phase": dbg.get("phase"),
                "last_event_human": dbg.get("last_event_human"),
                "last_error": dbg.get("last_error"),
                "thread_alive": _alive,
            })
            if st.button("🧪 Realtime-Smoketest"):
                try:
                    ok, msg = asyncio.run(_realtime_smoketest())
                except RuntimeError:
                    # If an event loop is already running (e.g., some environments), use a fallback
                    ok, msg = False, "Smoketest konnte nicht ausgeführt werden (laufender Event-Loop)."
                except Exception as e:
                    ok, msg = False, f"Smoketest unerwarteter Fehler: {type(e).__name__}: {e}"
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
            if st.button("🔁 Realtime neu starten"):
                # Stop & restart worker
                st.session_state["_rt_started"] = False
                st.session_state.pop("_rt_thread", None)
                _RT_FLAGS.update({
                    "client_created": False,
                    "channel_subscribed": False,
                    "connect_ok": False,
                    "last_error": None,
                    "phase": "starting",
                })
                _ensure_realtime_started()
                st.success("Realtime neu gestartet")
                st.rerun()
            st.caption("Falls `channel_subscribed` oder `connect_ok` False sind, prüfe die Publication `supabase_realtime` im Supabase-Dashboard und die Paketversion.")

        st.info("Statistik & Account – folgt. Hier kommen Profile, Verlauf, Einstellungen.")
        
        if st.button("🚪 Logout"):
            st.session_state.pop("user", None)
            st.session_state.pop("editing", None)
            if "user" in st.query_params:
                del st.query_params["user"]
            if "token" in st.query_params:
                del st.query_params["token"]
            st.rerun()
# endregion
