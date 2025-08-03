# region Imports
import streamlit as st
from supabase import create_client, Client
import pandas as pd
import bcrypt
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import time

# region UI Helpers (editing state)
# region UI Helpers (editing state)
def _set_editing_true():
    st.session_state["editing"] = True
def _set_editing_false():
    st.session_state["editing"] = False
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

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
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


def fetch_pending_for_user(user: str):
    # pending where user participates
    res = supabase.table("pending_matches").select("*").or_(f"a.eq.{user},b.eq.{user}").order("datum", desc=True).execute()
    rows = res.data or []
    to_confirm = []
    waiting_opponent = []
    for r in rows:
        if r["a"] == user and not r.get("confa", False):
            to_confirm.append(r)
        elif r["b"] == user and not r.get("confb", False):
            to_confirm.append(r)
        elif (r["a"] == user and r.get("confa", False) and not r.get("confb", False)) or \
             (r["b"] == user and r.get("confb", False) and not r.get("confa", False)):
            waiting_opponent.append(r)
    return to_confirm, waiting_opponent


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
        # if user hasn't confirmed yet
        needs_a = not r.get("confa", False)
        needs_b = not r.get("confb", False)
        if needs_a or needs_b:
            to_confirm.append(r)
        elif r.get("confa", False) and not r.get("confb", False):
            waiting.append(r)
    return to_confirm, waiting


def submit_round_pending(creator: str, teilnehmer: list[str], finalisten: tuple[str|None,str|None], sieger: str):
    teilnehmer = [t for t in (teilnehmer or []) if t]
    if len(teilnehmer) < 3:
        return False, "Mindestens 3 Teilnehmer erforderlich."
    if sieger not in teilnehmer:
        return False, "Sieger muss Teilnehmer sein."
    f1, f2 = finalisten if isinstance(finalisten, tuple) else (None, None)
    for f in (f1, f2):
        if f and f not in teilnehmer:
            return False, "Finalisten müssen Teilnehmer sein."
    confa = creator in teilnehmer
    confb = False
    supabase.table("pending_rounds").insert({
        "datum": now_utc_iso(),
        "teilnehmer": ";".join(teilnehmer),
        "finalisten": ";".join([x for x in (f1, f2) if x]),
        "sieger": sieger,
        "confa": confa, "confb": confb,
    }).execute()
    try:
        st.cache_data.clear()
    except Exception:
        pass
    return True, "Rundlauf eingereicht. Warte auf Bestätigung."


def confirm_pending_round(row_id: str, user: str):
    r = supabase.table("pending_rounds").select("*").eq("id", row_id).single().execute().data
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
    st.header(f"👋 Willkommen, {st.session_state.user}!")
    
    st.session_state.setdefault("editing", False)

    main_tab1, main_tab2, main_tab3 = st.tabs(["Willkommen", "Spielen", "Account"])

    with main_tab1:
        # Willkommen: ELO Anzeige, Matches bestätigen, letzte 5 Spiele
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
                if c3.button("Bestätigen", key=f"w_conf_{it['mode']}_{it['id']}"):
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
                if c4.button("Ablehnen", key=f"w_rej_{it['mode']}_{it['id']}"):
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
        else:
            st.caption("Nichts zu bestätigen.")

        st.subheader("🕔 Vom Gegner ausstehend")
        if wait_all:
            for it in wait_all:
                c1, c2, c3 = st.columns([5,2,2])
                c1.write(f"[{it['mode']}] {it['text']}")
                c2.write(fmt_dt_local(it['datum']))
                if c3.button("Ablehnen", key=f"w_rej_wait_{it['mode']}_{it['id']}"):
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
        else:
            st.caption("Keine offenen Anfragen beim Gegner.")

        st.subheader("Letzte Spiele")
        last = load_last_matches(5)
        if last:
            df_last = pd.DataFrame(last)
            if not df_last.empty and "datum" in df_last.columns:
                df_last["datum"] = pd.to_datetime(df_last["datum"], utc=True).dt.tz_convert(BERLIN).dt.strftime("%d.%m.%Y %H:%M")
            st.dataframe(df_last)
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
                c1, c2 = st.columns(2)
                with c1:
                    a = st.selectbox("Spieler A", names, key="ein_a", on_change=_set_editing_true)
                    pa = st.number_input("Punkte A", min_value=0, step=1, key="ein_pa", on_change=_set_editing_true)
                with c2:
                    b = st.selectbox("Spieler B", names, index=1 if len(names)>1 else 0, key="ein_b", on_change=_set_editing_true)
                    pb = st.number_input("Punkte B", min_value=0, step=1, key="ein_pb", on_change=_set_editing_true)
                if st.button("✅ Bestätigen"):
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
                        _set_editing_false()
            else:
                st.info("Mindestens zwei Spieler erforderlich.")

        with sub2:
            st.markdown("### Doppel eintragen")
            data_players = supabase.table("players").select("name").order("name").execute().data
            names = [p["name"] for p in data_players] if data_players else []
            if len(names) >= 4:
                c1, c2 = st.columns(2)
                with c1:
                    a1 = st.selectbox("Team A – Spieler 1", names, key="d_a1", on_change=_set_editing_true)
                    a2 = st.selectbox("Team A – Spieler 2", names, key="d_a2", on_change=_set_editing_true)
                    pa = st.number_input("Punkte Team A", min_value=0, step=1, key="d_pa", on_change=_set_editing_true)
                with c2:
                    b1 = st.selectbox("Team B – Spieler 1", names, key="d_b1", on_change=_set_editing_true)
                    b2 = st.selectbox("Team B – Spieler 2", names, key="d_b2", on_change=_set_editing_true)
                    pb = st.number_input("Punkte Team B", min_value=0, step=1, key="d_pb", on_change=_set_editing_true)
                if st.button("✅ Doppel einreichen"):
                    ok, msg = submit_double_pending(st.session_state.user, a1, a2, b1, b2, int(pa), int(pb))
                    if ok:
                        st.success("Doppel angelegt")
                    else:
                        st.error(msg)
                    _set_editing_false()
            else:
                st.info("Mindestens vier Spieler erforderlich.")

        with sub3:
            st.markdown("### Rundlauf eintragen")
            data_players = supabase.table("players").select("name").order("name").execute().data
            names = [p["name"] for p in data_players] if data_players else []
            if len(names) >= 3:
                participants = st.multiselect("Teilnehmer", names, key="r_parts", on_change=_set_editing_true)
                winner = st.selectbox("Sieger", participants if participants else [""], key="r_win", on_change=_set_editing_true)
                fin_cols = st.columns(2)
                with fin_cols[0]:
                    fin1 = st.selectbox("Finalist 1 (optional)", [""] + participants, key="r_f1", on_change=_set_editing_true)
                with fin_cols[1]:
                    fin2 = st.selectbox("Finalist 2 (optional)", [""] + participants, key="r_f2", on_change=_set_editing_true)
                if st.button("✅ Rundlauf einreichen"):
                    f1 = fin1 if fin1 else None
                    f2 = fin2 if fin2 else None
                    ok, msg = submit_round_pending(st.session_state.user, participants, (f1, f2), winner)
                    if ok:
                        st.success("Rundlauf angelegt")
                    else:
                        st.error(msg)
                    _set_editing_false()
            else:
                st.info("Mindestens drei Spieler erforderlich.")

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
        else:
            st.caption("Keine offenen Anfragen beim Gegner.")

    with main_tab3:
        st.info("Statistik & Account – folgt. Hier kommen Profile, Verlauf, Einstellungen.")
        
        if st.button("🚪 Logout"):
            st.session_state.pop("user", None)
            st.session_state.pop("editing", None)
            if "user" in st.query_params:
                del st.query_params["user"]
            if "token" in st.query_params:
                del st.query_params["token"]
            st.rerun()
        
        # Auto-refresh loop (only when logged in)
        if not st.session_state.get("editing", False):
            time.sleep(20)
            st.rerun()
# endregion
