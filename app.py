# region imports
from __future__ import annotations
import streamlit as st
import pandas as pd
from zoneinfo import ZoneInfo
from typing import Optional
import bcrypt
import uuid
from supabase import create_client
# endregion

# region app_setup
st.set_page_config(page_title="AK-Tischtennis", page_icon="🏓", layout="wide")
TZ = ZoneInfo("Europe/Berlin")
# endregion

# region supabase
@st.cache_resource
def get_supabase():
    """Erzeugt den Supabase-Client aus den Streamlit-Secrets.
    Schlägt mit einer Exception fehl, wenn Secrets fehlen oder das Paket nicht installiert ist.
    """
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

sp = get_supabase()
# endregion

# region data_loading
@st.cache_data(ttl=30)
def load_table(table_name: str) -> pd.DataFrame:
    """Lädt eine Supabase-Tabelle vollständig in ein DataFrame.
    - Spaltennamen -> lower()
    - 'datum' -> Europe/Berlin
    - Bei Fehler (z. B. Tabelle existiert nicht) leeres DF
    """
    try:
        res = sp.table(table_name).select("*").execute()
        data = res.data or []
    except Exception:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df.columns = [str(c).lower() for c in df.columns]
    if "datum" in df.columns:
        df["datum"] = pd.to_datetime(df["datum"], errors="coerce", utc=True).dt.tz_convert(TZ)
    return df

def clear_table_cache():
    load_table.clear()
# endregion

# region auth_helpers
def hash_pin(pin: str) -> str:
    return bcrypt.hashpw(pin.encode(), bcrypt.gensalt()).decode()

def check_pin(entered: str, stored: str) -> bool:
    """Vergleicht eingegebene PIN mit gespeichertem Wert (unterstützt Legacy-Klartext)."""
    if stored and (stored.startswith("$2b$") or stored.startswith("$2a$")):
        try:
            return bcrypt.checkpw(entered.encode(), stored.encode())
        except Exception:
            return False
    return entered == (stored or "")

def norm_name(s: str) -> str:
    """Normalisiert Namen für Vergleiche: entfernt alle Whitespaces und macht lowercase."""
    if s is None:
        return ""
    return "".join(str(s).split()).lower()

def find_player_by_name_normalized(name: str):
    """Sucht einen Spieler, wobei Groß/Kleinschreibung und Leerzeichen ignoriert werden."""
    target = norm_name(name)
    try:
        res = sp.table("players").select("id,name,pin,auto_token").execute()
        for rec in (res.data or []):
            if norm_name(rec.get("name", "")) == target:
                return rec
    except Exception:
        pass
    return None

def try_auto_login_from_query():
    """Auto-Login via URL-Parameter ?user=&token= (wenn players.auto_token vorhanden).
    Greift nur, wenn noch nicht eingeloggt.
    """
    if st.session_state.get("logged_in"):
        return
    q = st.query_params
    if "user" not in q or "token" not in q:
        return
    user = q.get("user")
    token = q.get("token")
    rec = find_player_by_name_normalized(user)
    if not rec:
        return
    if str(rec.get("auto_token", "")) == str(token):
        st.session_state.logged_in = True
        st.session_state.player_id = rec.get("id")
        st.session_state.player_name = rec.get("name")
        # Entferne sensible Query-Parameter aus der URL und lade sauber neu
        try:
            st.query_params.clear()
            st.rerun()
        except Exception:
            pass

# endregion

# region player_helpers

def get_current_user() -> dict | None:
    """Lädt den aktuell eingeloggten Spieler-Datensatz frisch aus Supabase."""
    try:
        pid = st.session_state.get("player_id")
        if not pid:
            return None
        rec = sp.table("players").select("*").eq("id", pid).single().execute().data
        return rec
    except Exception:
        return None

@st.cache_data(ttl=30)
def get_player_maps():
    """Gibt (id_to_name, name_to_id) basierend auf der players-Tabelle zurück."""
    df = load_table("players")
    id_to_name = {}
    name_to_id = {}
    if not df.empty:
        for _, r in df.iterrows():
            pid = str(r.get("id"))
            nm = str(r.get("name"))
            if pid and nm:
                id_to_name[pid] = nm
                name_to_id[nm] = pid
    return id_to_name, name_to_id


@st.cache_data(ttl=30)
def table_has_creator(table: str) -> bool:
    """Prüft, ob die Pending-Tabelle eine Spalte 'creator' hat."""
    try:
        sp.table(table).select("creator").limit(1).execute()
        return True
    except Exception:
        return False

# endregion

# region game_helpers
from datetime import datetime

def _utc_iso(ts) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").isoformat()


# --- ELO: Utilities (aus alter Datei adaptiert) ---


def _compute_gelo_from_parts(elo: int, d_elo: int, r_elo: int) -> int:
    # 0.6 Einzel, 0.25 Doppel, 0.15 Rundlauf
    return int(round(0.6 * int(elo) + 0.25 * int(d_elo) + 0.15 * int(r_elo)))


# --- Dynamische G-ELO-Berechnung nach Spielanteilen ---
def _compute_gelo_dynamic(elo:int, d_elo:int, r_elo:int, n_e:int, n_d:int, n_r:int) -> int:
    """Berechnet G‑ELO nach Spielanteilen.
    Gewichte = Anteil der jeweiligen Spielanzahl; spielt jemand einen Modus nicht, zählt er 0.
    """
    total = n_e + n_d + n_r
    if total == 0:
        return int(round((elo + d_elo + r_elo)/3))  # Fallback, sollte nie passieren
    w_e, w_d, w_r = n_e/total, n_d/total, n_r/total
    return int(round(w_e*elo + w_d*d_elo + w_r*r_elo))


def calc_elo(r_a: float, r_b: float, score_a: float, k: float = 64) -> int:
    """ELO-Formel mit Erwartungswert; k wird außerhalb ggf. um Margenfaktor skaliert."""
    exp_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    return int(round(r_a + k * (score_a - exp_a)))


def calc_doppel_elo(r1: float, r2: float, opp_avg: float, s: float, k: float = 48) -> tuple[int, int]:
    team_avg = (r1 + r2) / 2
    exp = 1 / (1 + 10 ** ((opp_avg - team_avg) / 400))
    delta = k * (s - exp)
    return int(round(r1 + delta)), int(round(r2 + delta))

# --- Rundlauf-ELO ---
def calc_round_elo(r: float, avg: float, s: float, k: int = 48) -> int:
    """Rundlauf-ELO: Sieger=1, Zweiter=0.5, andere=0."""
    exp = 1 / (1 + 10 ** ((avg - r) / 400))
    return int(round(r + k * (s - exp)))

def update_round_after_confirm_id(participant_ids: list[str], fin1_id: str, fin2_id: str, winner_id: str, k: int = 48) -> None:
    """Aktualisiert r_elo/Stats und g_elo aller Teilnehmer eines Rundlaufs (ID-basiert)."""
    # 1) Aktuelle Werte holen
    current = {}
    for pid in participant_ids:
        rec = sp.table("players").select("id, name, r_elo, r_siege, r_zweite, r_niederlagen, r_spiele, elo, spiele, d_spiele, d_elo").eq("id", pid).single().execute().data
        if rec:
            current[str(pid)] = rec
    if not current:
        return
    # 2) Durchschnitt berechnen (auf Basis r_elo vor Update)
    avg = float(pd.Series([float(v.get("r_elo", 1200)) for v in current.values()]).mean())
    # 3) Für jeden Spieler neues r_elo + Stats
    for pid, rec in current.items():
        old_r = float(rec.get("r_elo", 1200))
        if pid == str(winner_id):
            s = 1.0
            inc = {"r_siege": 1, "r_zweite": 0, "r_niederlagen": 0}
        elif pid in (str(fin1_id), str(fin2_id)):
            # Wenn nicht Gewinner, aber Finalist → Zweiter
            if pid != str(winner_id):
                s = 0.5
                inc = {"r_siege": 0, "r_zweite": 1, "r_niederlagen": 0}
            else:
                s = 1.0
                inc = {"r_siege": 1, "r_zweite": 0, "r_niederlagen": 0}
        else:
            s = 0.0
            inc = {"r_siege": 0, "r_zweite": 0, "r_niederlagen": 1}
        new_r = calc_round_elo(old_r, avg, s, k)
        payload = {
            "r_elo": int(new_r),
            "r_siege": int(rec.get("r_siege", 0)) + inc["r_siege"],
            "r_zweite": int(rec.get("r_zweite", 0)) + inc["r_zweite"],
            "r_niederlagen": int(rec.get("r_niederlagen", 0)) + inc["r_niederlagen"],
            "r_spiele": int(rec.get("r_spiele", 0)) + 1,
        }
        payload["g_elo"] = _compute_gelo_dynamic(
            int(rec.get("elo", 1200)),
            int(rec.get("d_elo", 1200)),
            payload["r_elo"],
            int(rec.get("spiele", 0)),
            int(rec.get("d_spiele", 0)),
            payload["r_spiele"],
        )
        sp.table("players").update(payload).eq("id", pid).execute()


def update_single_after_confirm_id(a_id: str, b_id: str, pa: int, pb: int, k_base: int = 64) -> None:
    if int(pa) == int(pb):
        return
    a = sp.table("players").select("id, elo, siege, niederlagen, spiele, d_spiele, r_spiele, d_elo, r_elo").eq("id", a_id).single().execute().data
    b = sp.table("players").select("id, elo, siege, niederlagen, spiele, d_spiele, r_spiele, d_elo, r_elo").eq("id", b_id).single().execute().data
    if not a or not b:
        return
    r_a, r_b = float(a.get("elo", 1200)), float(b.get("elo", 1200))
    margin = abs(int(pa) - int(pb))  # 0–11
    k_eff = k_base * (1 + margin / 11)
    winner_is_a = int(pa) > int(pb)
    new_r_a = calc_elo(r_a, r_b, 1 if winner_is_a else 0, k_eff)
    new_r_b = calc_elo(r_b, r_a, 0 if winner_is_a else 1, k_eff)
    a_payload = {
        "elo": new_r_a,
        "siege": int(a.get("siege", 0)) + (1 if winner_is_a else 0),
        "niederlagen": int(a.get("niederlagen", 0)) + (0 if winner_is_a else 1),
        "spiele": int(a.get("spiele", 0)) + 1,
    }
    b_payload = {
        "elo": new_r_b,
        "siege": int(b.get("siege", 0)) + (0 if winner_is_a else 1),
        "niederlagen": int(b.get("niederlagen", 0)) + (1 if winner_is_a else 0),
        "spiele": int(b.get("spiele", 0)) + 1,
    }
    a_payload["g_elo"] = _compute_gelo_dynamic(
        a_payload["elo"],
        int(a.get("d_elo", 1200)),
        int(a.get("r_elo", 1200)),
        a_payload["spiele"],
        int(a.get("d_spiele", 0)),
        int(a.get("r_spiele", 0)),
    )
    b_payload["g_elo"] = _compute_gelo_dynamic(
        b_payload["elo"],
        int(b.get("d_elo", 1200)),
        int(b.get("r_elo", 1200)),
        b_payload["spiele"],
        int(b.get("d_spiele", 0)),
        int(b.get("r_spiele", 0)),
    )
    sp.table("players").update(a_payload).eq("id", a_id).execute()
    sp.table("players").update(b_payload).eq("id", b_id).execute()


def update_double_after_confirm_id(a1_id: str, a2_id: str, b1_id: str, b2_id: str, pa: int, pb: int, k_base: int = 48) -> None:
    if int(pa) == int(pb):
        return
    A1 = sp.table("players").select("id, d_elo, d_siege, d_niederlagen, d_spiele, elo, spiele, r_spiele, r_elo").eq("id", a1_id).single().execute().data
    A2 = sp.table("players").select("id, d_elo, d_siege, d_niederlagen, d_spiele, elo, spiele, r_spiele, r_elo").eq("id", a2_id).single().execute().data
    B1 = sp.table("players").select("id, d_elo, d_siege, d_niederlagen, d_spiele, elo, spiele, r_spiele, r_elo").eq("id", b1_id).single().execute().data
    B2 = sp.table("players").select("id, d_elo, d_siege, d_niederlagen, d_spiele, elo, spiele, r_spiele, r_elo").eq("id", b2_id).single().execute().data
    if not A1 or not A2 or not B1 or not B2:
        return
    ra1, ra2 = float(A1.get("d_elo", 1200)), float(A2.get("d_elo", 1200))
    rb1, rb2 = float(B1.get("d_elo", 1200)), float(B2.get("d_elo", 1200))
    a_avg, b_avg = (ra1 + ra2) / 2, (rb1 + rb2) / 2
    margin = abs(int(pa) - int(pb))
    k_eff = k_base * (1 + margin / 11)
    team_a_win = 1 if int(pa) > int(pb) else 0
    nr1, nr2 = calc_doppel_elo(ra1, ra2, b_avg, team_a_win, k_eff)
    nr3, nr4 = calc_doppel_elo(rb1, rb2, a_avg, 1 - team_a_win, k_eff)
    updates = [
        (a1_id, A1, nr1, team_a_win),
        (a2_id, A2, nr2, team_a_win),
        (b1_id, B1, nr3, 1 - team_a_win),
        (b2_id, B2, nr4, 1 - team_a_win),
    ]
    for pid, Rec, new_val, win in updates:
        payload = {
            "d_elo": int(new_val),
            "d_siege": int(Rec.get("d_siege", 0)) + int(win),
            "d_niederlagen": int(Rec.get("d_niederlagen", 0)) + (1 - int(win)),
            "d_spiele": int(Rec.get("d_spiele", 0)) + 1,
        }
        payload["g_elo"] = _compute_gelo_dynamic(
            int(Rec.get("elo", 1200)),
            payload["d_elo"],
            int(Rec.get("r_elo", 1200)),
            int(Rec.get("spiele", 0)),
            payload["d_spiele"],
            int(Rec.get("r_spiele", 0)),
        )
        sp.table("players").update(payload).eq("id", pid).execute()


# --- Neue create_pending_* Funktionen mit optionaler creator-Spalte ---

def create_pending_single(creator_id: str, opponent_id: str, s_a: int, s_b: int, a_id: str | None = None):
    """Erstellt ein Einzel-Pending.
    - Wenn a_id None ist, spielt der Ersteller (a=creator_id) gegen opponent_id (b).
    - Wenn a_id gesetzt ist, wird a=a_id und b=opponent_id; der Ersteller kann außen vor bleiben.
    """
    a_val = creator_id if a_id is None else a_id
    payload = {
        "datum": _utc_iso(pd.Timestamp.now(tz=TZ)),
        "a": a_val, "b": opponent_id,
        "punktea": int(s_a), "punkteb": int(s_b),
        "confa": False, "confb": False,
    }
    if table_has_creator("pending_matches"):
        payload["creator"] = creator_id
    sp.table("pending_matches").insert(payload).execute()


def create_pending_double(creator_id: str, partner_id: str, opp1_id: str, opp2_id: str, team_a_is_creator: bool = True):
    """Erstellt ein Doppel-Pending.
    - team_a_is_creator=True: Team A = (creator_id, partner_id)
    - team_a_is_creator=False: Team A = (partner_id, opp1_id) und Team B = (opp2_id, creator_id) wird NICHT automatisch gesetzt; stattdessen erstellt der Ersteller nur für andere (A=partner_id+opp1_id, B=opp2_id+<zweiter Gegner>). In diesem Fall muss partner_id bereits beide Team-A-Spieler enthalten, daher verwenden wir partner_id als A2 und erwarten, dass opp1_id/opp2_id die Team‑B‑Spieler sind.
    """
    if team_a_is_creator:
        a1, a2, b1, b2 = creator_id, partner_id, opp1_id, opp2_id
    else:
        # Ersteller spielt nicht mit → A = (partner_id, opp1_id), B = (opp2_id, irgendein anderer). Für UI-Varianten bauen wir unten einen zweiten Aufruf.
        a1, a2, b1, b2 = partner_id, opp1_id, opp2_id, None  # b2 wird in der UI-Logik gesetzt
    payload = {
        "datum": _utc_iso(pd.Timestamp.now(tz=TZ)),
        "a1": a1, "a2": a2, "b1": b1, "b2": b2,
        "punktea": 0, "punkteb": 0,  # Scores werden aus der UI-Variante gesetzt
        "confa": False, "confb": False,
    }
    if table_has_creator("pending_doubles"):
        payload["creator"] = creator_id
    # Insert erfolgt in der UI mit finalen Werten (wir überschreiben punktea/punkteb/b2 dort vor dem Insert)
    return payload


def create_pending_round(creator_id: str, participant_ids: list[str], fin1_id: str, fin2_id: str, winner_id: str):
    """Erstellt ein Rundlauf-Pending mit Teilnehmern + Finalisten (1. und 2.) + Sieger."""
    teilnehmer = ";".join([pid for pid in participant_ids if pid])
    finalisten = ";".join([pid for pid in [fin1_id, fin2_id] if pid])
    payload = {
        "datum": _utc_iso(pd.Timestamp.now(tz=TZ)),
        "teilnehmer": teilnehmer,
        "finalisten": finalisten,
        "sieger": winner_id,
        "confa": False, "confb": False,
    }
    if table_has_creator("pending_rounds"):
        payload["creator"] = creator_id
    sp.table("pending_rounds").insert(payload).execute()

# --- Confirm / Reject (ohne ELO-Update; das bauen wir im nächsten Schritt) ---

def confirm_pending_single(row: pd.Series):
    # Update ELO/Stats (IDs)
    update_single_after_confirm_id(str(row["a"]), str(row["b"]), int(row["punktea"]), int(row["punkteb"]))
    # Persist match
    sp.table("matches").insert({
        "datum": _utc_iso(pd.Timestamp(row["datum"])),
        "a": row["a"], "b": row["b"],
        "punktea": int(row["punktea"]), "punkteb": int(row["punkteb"]),
    }).execute()
    # Remove pending
    sp.table("pending_matches").delete().eq("id", row["id"]).execute()


def confirm_pending_double(row: pd.Series):
    # Update ELO/Stats (IDs)
    update_double_after_confirm_id(str(row["a1"]), str(row["a2"]), str(row["b1"]), str(row["b2"]), int(row["punktea"]), int(row["punkteb"]))
    # Persist match
    sp.table("doubles").insert({
        "datum": _utc_iso(pd.Timestamp(row["datum"])),
        "a1": row["a1"], "a2": row["a2"],
        "b1": row["b1"], "b2": row["b2"],
        "punktea": int(row["punktea"]), "punkteb": int(row["punkteb"]),
    }).execute()
    # Remove pending
    sp.table("pending_doubles").delete().eq("id", row["id"]).execute()


def confirm_pending_round(row: pd.Series):
    # IDs aus Strings extrahieren
    participant_ids = [pid for pid in str(row["teilnehmer"]).split(";") if pid]
    fin_list = [pid for pid in str(row.get("finalisten") or "").split(";") if pid]
    fin1_id = fin_list[0] if len(fin_list) > 0 else None
    fin2_id = fin_list[1] if len(fin_list) > 1 else None
    winner_id = str(row.get("sieger")) if row.get("sieger") else None
    # ELO/Stats updaten (nur wenn alles vorhanden)
    if participant_ids and fin1_id and fin2_id and winner_id:
        update_round_after_confirm_id(participant_ids, fin1_id, fin2_id, winner_id)
    # Persist in rounds
    sp.table("rounds").insert({
        "datum": _utc_iso(pd.Timestamp(row["datum"])),
        "teilnehmer": row["teilnehmer"],
        "finalisten": row.get("finalisten"),
        "sieger": row.get("sieger"),
    }).execute()
    # Remove pending
    sp.table("pending_rounds").delete().eq("id", row["id"]).execute()


def reject_pending(table: str, id_: str):
    sp.table(table).delete().eq("id", id_).execute()

# endregion

# region login_ui
def login_register_ui():
    """UI für Login & Registrierung (PIN-basiert) inkl. "Angemeldet bleiben".
    Bei erstem Login aus Legacy-Klartext wird die PIN auf bcrypt migriert.
    """
    tabs = st.tabs(["Einloggen", "Registrieren"])    

    with tabs[0]:
        name = st.text_input("Spielername", key="login_name")
        pin = st.text_input("PIN", type="password", key="login_pin")
        remember = st.checkbox("Angemeldet bleiben")
        if st.button("Einloggen", type="primary"):
            # Spieler holen
            rec = find_player_by_name_normalized(name)
            if not rec:
                st.error("Spielername nicht gefunden.")
                return
            stored_pin = str(rec.get("pin", ""))
            if not check_pin(pin, stored_pin):
                st.error("PIN falsch.")
                return
            # ggf. Legacy -> bcrypt upgraden
            if not (stored_pin.startswith("$2b$") or stored_pin.startswith("$2a$")):
                try:
                    sp.table("players").update({"pin": hash_pin(pin)}).eq("id", rec["id"]).execute()
                except Exception:
                    pass
            # Session setzen
            st.session_state.logged_in = True
            st.session_state.player_id = rec.get("id")
            st.session_state.player_name = rec.get("name")

            # Remember-Me via auto_token + URL-Params
            if remember:
                try:
                    token = uuid.uuid4().hex
                    # Versuch: Spalte auto_token aktualisieren (falls nicht vorhanden, wird es fehlschlagen)
                    sp.table("players").update({"auto_token": token}).eq("id", rec["id"]).execute()
                    st.query_params.update({"user": rec.get("name"), "token": token})
                except Exception:
                    st.info("Hinweis: 'auto_token' Spalte nicht vorhanden – automatischer Login per URL-Token ist deaktiviert.")
            st.success(f"Willkommen, {rec.get('name')}!")
            st.rerun()

    with tabs[1]:
        r_name = st.text_input("Neuer Spielername", key="reg_name")
        r_pin1 = st.text_input("PIN wählen (4-stellig)", type="password", key="reg_pin1")
        r_pin2 = st.text_input("PIN bestätigen", type="password", key="reg_pin2")
        if st.button("Registrieren"):
            if not r_name or not r_pin1 or not r_pin2:
                st.warning("Name und PIN eingeben.")
                return
            if r_pin1 != r_pin2:
                st.warning("PINs stimmen nicht überein.")
                return
            # Kollisionen ignorieren Groß/Kleinschreibung und Leerzeichen
            try:
                rows = sp.table("players").select("id,name").execute().data or []
            except Exception:
                rows = []
            if any(norm_name(row.get("name")) == norm_name(r_name) for row in rows):
                st.warning("Spieler existiert bereits (Groß/Kleinschreibung/Leerzeichen ignoriert).")
                return
            payload = {
                "name": r_name,
                "pin": hash_pin(r_pin1),
                # Stats-Defaults, falls Spalten existieren
                "elo": 1200, "siege": 0, "niederlagen": 0, "spiele": 0,
                "d_elo": 1200, "d_siege": 0, "d_niederlagen": 0, "d_spiele": 0,
                "r_elo": 1200, "r_siege": 0, "r_zweite": 0, "r_niederlagen": 0, "r_spiele": 0,
                "g_elo": 1200,
            }
            try:
                sp.table("players").insert(payload).execute()
                clear_table_cache()
                st.success("Registriert! Bitte einloggen.")
            except Exception as e:
                st.error(f"Konnte nicht registrieren: {e}")
# endregion

# region layout_header
st.header("🏓 AK-Tischtennis")
st.markdown(
    """
    <style>
      /* Make all Streamlit tab bars span full width and distribute tabs evenly */
      div.stTabs > div[role="tablist"],
      div.stTabs [data-baseweb="tab-list"] {
        display: flex !important;
        width: 100% !important;
      }
      div.stTabs > div[role="tablist"] > div[role="tab"],
      div.stTabs [data-baseweb="tab-list"] [data-baseweb="tab"] {
        flex: 1 1 0 !important;
        justify-content: center !important;
        text-align: center !important;
      }
      /* Optional: reduce default gaps for tighter fit on mobile */
      div.stTabs [data-baseweb="tab-list"] { gap: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)
# endregion

# region session_init
# Session-Defaults setzen und Auto-Login via URL prüfen
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.player_id = None
    st.session_state.player_name = None

# Versuch: Auto-Login direkt beim Start
try_auto_login_from_query()
# endregion

# region ui_logged_in

def _metric_val(user: dict, key: str, default: int = 1200) -> int:
    try:
        return int(user.get(key, default))
    except Exception:
        return default

def logged_in_header(user: dict):
    name = user.get('name', 'Spieler')
    gelo = _metric_val(user, 'g_elo')
    elo = _metric_val(user, 'elo')
    d_elo = _metric_val(user, 'd_elo')
    r_elo = _metric_val(user, 'r_elo')

    st.markdown(
        f"""
        <style>
        .elo-wrap {{ margin: 0.25rem 0 0.5rem; }}
        .welcome {{ font-size: 1.25rem; font-weight: 600; margin: 0 0 .25rem; }}
        .elo-center {{ text-align: center; margin: .25rem 0 .5rem; }}
        .elo-center .value {{ font-size: 40px; font-weight: 700; line-height: 1; }}
        .elo-grid {{ display: flex; gap: 8px; justify-content: center; align-items: stretch; flex-wrap: nowrap; }}
        .elo-card {{ flex: 0 1 33%; max-width: 33%; padding: 8px 10px; border: 1px solid rgba(255,255,255,.15); border-radius: 10px; text-align: center; backdrop-filter: blur(2px); }}
        .elo-card .label {{ font-size: 12px; opacity: .75; margin-bottom: 2px; }}
        .elo-card .value {{ font-size: 18px; font-weight: 600; line-height: 1.1; }}
        @media (max-width: 480px) {{
          .elo-center .value {{ font-size: 34px; }}
          .elo-grid {{ gap: 6px; }}
          .elo-card {{ padding: 6px 6px; }}
          .elo-card .value {{ font-size: 16px; }}
        }}
        </style>
        <div class="elo-wrap">
          <div class="welcome">Willkommen, {name}</div>
          <div class="elo-center"><div class="value">{gelo}</div></div>
          <div class="elo-grid">
              <div class="elo-card"><div class="label">Einzel</div><div class="value">{elo}</div></div>
              <div class="elo-card"><div class="label">Doppel</div><div class="value">{d_elo}</div></div>
              <div class="elo-card"><div class="label">Rundlauf</div><div class="value">{r_elo}</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_round_vs_card(r: pd.Series, id_to_name: dict, *, highlight_name: str = "", key: str = "", on_reject=None, button_label: str = "❌ Ablehnen"):
    """Rendert eine Rundlauf-Karte als Expander – **nur** mit Titel + Ablehnen-Button."""
    teiln_ids = [pid for pid in str(r.get("teilnehmer") or "").split(";") if pid]
    teiln = [id_to_name.get(pid, pid) for pid in teiln_ids]
    fin_list = [pid for pid in str(r.get("finalisten") or "").split(";") if pid]
    winner_id = str(r.get("sieger")) if r.get("sieger") else None
    winner_n = id_to_name.get(winner_id, winner_id) or "-"
    second_id = fin_list[1] if len(fin_list) > 1 and fin_list[0] == winner_id else (fin_list[0] if len(fin_list) > 0 else None)
    second_n = id_to_name.get(second_id, second_id) if second_id else "-"

    # Expander-Titel (ohne Modus, Medaillen vor den Namen)
    title = f"{', '.join(teiln)} – 🥇 {winner_n}, 🥈 {second_n}"
    exp = st.expander(title, expanded=False)

    with exp:
        if st.button(button_label, key=key):
            if on_reject:
                on_reject(r["id"])  # erwartet Tabellen-ID


# --- Helper: Einzel-Karte im VS-Layout (mit Sieger-Medaille) ---
def render_single_vs_card(r: pd.Series, id_to_name: dict, *, highlight_name: str = "", key: str = "", on_reject=None, button_label: str = "❌ Ablehnen"):
    a_n = id_to_name.get(str(r["a"]), r["a"]) ; b_n = id_to_name.get(str(r["b"]), r["b"])
    pa, pb = int(r.get("punktea", 0)), int(r.get("punkteb", 0))
    left_medal = "🥇 " if pa > pb else ""
    right_medal = "🥇 " if pb > pa else ""
    title = f"{left_medal}{a_n} - {pa}:{pb} - {right_medal}{b_n}"
    exp = st.expander(title, expanded=False)
    with exp:
        if st.button(button_label, key=key):
            if on_reject:
                on_reject(r["id"])  # erwartet Tabellen-ID


# --- Helper: Doppel-Karte im VS-Layout (mit Sieger-Medaille) ---
def render_double_vs_card(r: pd.Series, id_to_name: dict, *, highlight_name: str = "", key: str = "", on_reject=None, button_label: str = "❌ Ablehnen"):
    a1 = id_to_name.get(str(r["a1"]), r["a1"]) ; a2 = id_to_name.get(str(r["a2"]), r["a2"]) 
    b1 = id_to_name.get(str(r["b1"]), r["b1"]) ; b2 = id_to_name.get(str(r["b2"]), r["b2"]) 
    pa, pb = int(r.get("punktea", 0)), int(r.get("punkteb", 0))
    team_a = f"{a1}/{a2}"
    team_b = f"{b1}/{b2}"
    left_medal = "🥇 " if pa > pb else ""
    right_medal = "🥇 " if pb > pa else ""
    title = f"{left_medal}{team_a} - {pa}:{pb} - {right_medal}{team_b}"
    exp = st.expander(title, expanded=False)
    with exp:
        if st.button(button_label, key=key):
            if on_reject:
                on_reject(r["id"])  # erwartet Tabellen-ID


def get_current_user() -> dict | None:
    try:
        pid = st.session_state.get("player_id")
        if not pid:
            return None
        rec = sp.table("players").select("*").eq("id", pid).single().execute().data
        return rec
    except Exception:
        return None


def logged_in_ui():
    user = get_current_user()
    if not user:
        st.error("Nutzer nicht gefunden – bitte erneut einloggen.")
        for k in ("logged_in","player_id","player_name"):
            st.session_state.pop(k, None)
        st.rerun()
        return

    # Tabs: Übersicht (zuerst), Spielen, Account (Logout)
    tabs = st.tabs(["Übersicht", "Spielen", "Account"])  

    # region Übersicht
    with tabs[0]:
        logged_in_header(user)

        # --- Offene Bestätigungen (auch in Übersicht anzeigen) ---
        id_to_name, name_to_id = get_player_maps()
        me = st.session_state.get("player_id")

        st.divider()

        col_head, col_btn = st.columns([100,1])
        with col_head:
            st.markdown("### Offene Bestätigungen")
        with col_btn:
            if st.button("🔄", key="btn_refresh_confirmations_ovw"):
                clear_table_cache()
                st.rerun()
        
        pm = load_table("pending_matches")
        pdbl = load_table("pending_doubles")
        pr = load_table("pending_rounds")

        # --- Meine offenen Bestätigungen sammeln (pro Modus) ---
        info_rows_s, info_rows_d, info_rows_r = [], [], []
        # Einzel
        if not pm.empty:
            has_c = table_has_creator("pending_matches")
            if has_c:
                my_conf_s = pm[(pm["a"].astype(str).eq(str(me)) | pm["b"].astype(str).eq(str(me))) & (pm["creator"].astype(str) != str(me))]
            else:
                my_conf_s = pm[pm["b"].astype(str) == str(me)]
            for _, r in my_conf_s.iterrows():
                info_rows_s.append(r)
        # Doppel
        if not pdbl.empty:
            has_c_d = table_has_creator("pending_doubles")
            if has_c_d:
                part_mask = (pdbl[["a1","a2","b1","b2"]].astype(str) == str(me)).any(axis=1)
                my_conf_d = pdbl[part_mask & (pdbl["creator"].astype(str) != str(me))]
            else:
                my_conf_d = pdbl[(pdbl["a1"].astype(str) != str(me)) & ((pdbl["a2"].astype(str) == str(me)) | (pdbl["b1"].astype(str) == str(me)) | (pdbl["b2"].astype(str) == str(me)))]
            for _, r in my_conf_d.iterrows():
                info_rows_d.append(r)
        # Rundlauf
        if not pr.empty:
            has_c_r = table_has_creator("pending_rounds")
            if has_c_r:
                def _involved_not_creator(row):
                    teiln = [x for x in str(row.get("teilnehmer","")) .split(";") if x]
                    return (str(me) in teiln) and (str(row.get("creator")) != str(me))
                my_conf_r = pr[pr.apply(_involved_not_creator, axis=1)]
            else:
                def _is_involved_not_creator(row):
                    teiln = [x for x in str(row.get("teilnehmer","")) .split(";") if x]
                    return (str(me) in teiln) and (len(teiln) > 0 and teiln[0] != str(me))
                my_conf_r = pr[pr.apply(_is_involved_not_creator, axis=1)]
            for _, r in my_conf_r.iterrows():
                info_rows_r.append(r)

        # --- Global: Alle bestätigen (unter dem Refresh-Button) ---
        if any([info_rows_s, info_rows_d, info_rows_r]):
            if st.button("✅ Alle bestätigen", key="btn_accept_all_pending_ovw", type="primary"):
                try:
                    # Einzel
                    for r in info_rows_s:
                        confirm_pending_single(r)
                    # Doppel
                    for r in info_rows_d:
                        confirm_pending_double(r)
                    # Rundlauf
                    for r in info_rows_r:
                        confirm_pending_round(r)
                    clear_table_cache()
                    st.success("Alle bestätigbaren Spiele bestätigt.")
                    st.rerun()
                except Exception:
                    clear_table_cache()
                    st.warning("Massenbestätigung teilweise fehlgeschlagen. Seite neu laden und prüfen.")
        else:
            st.info("Keine offenen Bestätigungen.")

        # --- Karten-Ansicht der einzelnen Spiele (nur Ablehnen pro Karte) ---
        # Einzel-Karten
        me_name = user.get("name")
        for r in info_rows_s:
            render_single_vs_card(
                r, id_to_name,
                highlight_name=me_name,
                key=f"trej_s_{r['id']}_ovw",
                on_reject=lambda rid: (reject_pending("pending_matches", rid), clear_table_cache(), st.rerun()),
                button_label="❌ Ablehnen",
            )

        # Doppel-Karten
        me_name = user.get("name")
        for r in info_rows_d:
            render_double_vs_card(
                r, id_to_name,
                highlight_name=me_name,
                key=f"trej_d_{r['id']}_ovw",
                on_reject=lambda rid: (reject_pending("pending_doubles", rid), clear_table_cache(), st.rerun()),
                button_label="❌ Ablehnen",
            )

        # Rundlauf-Karten
        me_name = user.get("name")
        for r in info_rows_r:
            render_round_vs_card(
                r, id_to_name,
                highlight_name=me_name,
                key=f"trej_r_{r['id']}_ovw",
                on_reject=lambda rid: (reject_pending("pending_rounds", rid), clear_table_cache(), st.rerun()),
                button_label="❌ Ablehnen",
            )

        # --- Leaderboards & Letzte Spiele ---
        st.divider()
        st.markdown("### Leaderboards & Letzte Spiele")

        lb_tabs = st.tabs(["Gesamt", "Einzel", "Doppel", "Rundlauf", "Letzte Spiele"])

        # --- Helper: safe sorting and display of leaderboard ---
        def _show_lb(df_players: pd.DataFrame, col: str, title: str, highlight_name: str):
            if df_players.empty or col not in df_players.columns:
                st.info("Noch keine Daten.")
                return
            tmp = df_players[["name", col]].copy()
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce").fillna(0).astype(int)
            tmp = tmp.sort_values(col, ascending=False).reset_index(drop=True)
            tmp = tmp.rename(columns={"name": "Name", col: title})

            # Index praktisch unsichtbar machen: minimale Breite via Styler und transparente Farbe
            primary = st.get_option("theme.primaryColor") or "#dc2626"

            def _style_row(row: pd.Series):
                if str(row["Name"]) == str(highlight_name):
                    return [f"color: {primary}; font-weight: 700" for _ in row.index]
                return ["" for _ in row.index]

            sty = tmp.style.apply(_style_row, axis=1)
            # Erzwinge Zentrierung explizit pro Spalte (wirkt zuverlässiger als global)
            for _c in ["Name", title]:
                try:
                    sty = sty.set_properties(subset=[_c], **{"text-align": "center"})
                except Exception:
                    pass
            sty = sty.set_table_styles([
                {"selector": "th.row_heading", "props": [
                    ("width","1px"),("min-width","1px"),("max-width","1px"),
                    ("padding","0"),("border","none"),("overflow","hidden"),
                    ("color","transparent")
                ]},
                {"selector": "tbody th", "props": [
                    ("width","1px"),("min-width","1px"),("max-width","1px"),
                    ("padding","0"),("border","none"),("overflow","hidden"),
                    ("color","transparent")
                ]},
                {"selector": "th.blank", "props": [
                    ("width","1px"),("min-width","1px"),("max-width","1px"),
                    ("padding","0"),("border","none"),("overflow","hidden"),
                    ("color","transparent")
                ]},
                {"selector": "thead th", "props": [
                    ("text-align","center !important")
                ]},
                {"selector": "th.col_heading", "props": [
                    ("white-space","nowrap"), ("text-align","center !important")
                ]},
                {"selector": "tbody td", "props": [
                    ("text-align","center !important")
                ]},
            ], overwrite=False)
            st.table(sty)

        players_df = load_table("players")

        me_name = user.get("name")

        with lb_tabs[0]:  # Gesamt‑ELO
            _show_lb(players_df, "g_elo", "Gesamt‑ELO", me_name)
        with lb_tabs[1]:  # Einzel‑ELO
            _show_lb(players_df, "elo", "Einzel‑ELO", me_name)
        with lb_tabs[2]:  # Doppel‑ELO
            _show_lb(players_df, "d_elo", "Doppel‑ELO", me_name)
        with lb_tabs[3]:  # Rundlauf‑ELO
            _show_lb(players_df, "r_elo", "Rundlauf‑ELO", me_name)

        with lb_tabs[4]:  # Letzte Spiele
            m = load_table("matches")
            d = load_table("doubles")
            r = load_table("rounds")
            rows = []
            # Einzel
            if not m.empty:
                for _, x in m.iterrows():
                    a_n = id_to_name.get(str(x.get("a")), str(x.get("a")))
                    b_n = id_to_name.get(str(x.get("b")), str(x.get("b")))
                    rows.append({
                        "datum": x.get("datum"),
                        "Modus": "Einzel",
                        "Teilnehmer": f"{a_n} vs {b_n}",
                        "Ergebnis": f"{int(x.get('punktea',0))}:{int(x.get('punkteb',0))}",
                    })
            # Doppel
            if not d.empty:
                for _, x in d.iterrows():
                    a1 = id_to_name.get(str(x.get("a1")), str(x.get("a1")))
                    a2 = id_to_name.get(str(x.get("a2")), str(x.get("a2")))
                    b1 = id_to_name.get(str(x.get("b1")), str(x.get("b1")))
                    b2 = id_to_name.get(str(x.get("b2")), str(x.get("b2")))
                    rows.append({
                        "datum": x.get("datum"),
                        "Modus": "Doppel",
                        "Teilnehmer": f"{a1}/{a2} vs {b1}/{b2}",
                        "Ergebnis": f"{int(x.get('punktea',0))}:{int(x.get('punkteb',0))}",
                    })
            # Rundlauf
            if not r.empty:
                for _, x in r.iterrows():
                    teiln = [id_to_name.get(pid, pid) for pid in str(x.get("teilnehmer") or "").split(";") if pid]
                    fin_list = [id_to_name.get(pid, pid) for pid in str(x.get("finalisten") or "").split(";") if pid]
                    winner_n = id_to_name.get(str(x.get("sieger")), str(x.get("sieger")))
                    second_n = fin_list[1] if len(fin_list)>1 and fin_list[0]==winner_n else (fin_list[0] if len(fin_list)>0 else '-')
                    rows.append({
                        "datum": x.get("datum"),
                        "Modus": "Rundlauf",
                        "Teilnehmer": ", ".join(teiln),
                        "Ergebnis": f"1.: {winner_n}\n2.: {second_n}",
                    })
            if rows:
                df_last = pd.DataFrame(rows)
                df_last["datum"] = pd.to_datetime(df_last["datum"], errors="coerce")
                df_last = df_last.sort_values("datum", ascending=False, na_position="last").head(5)
                show_df = df_last [["Modus","Teilnehmer","Ergebnis"]].copy()
                primary = st.get_option("theme.primaryColor") or "#dc2626"

                def _style_last(row: pd.Series):
                    if me_name and str(me_name) in str(row["Teilnehmer"]):
                        return [f"color: {primary}; font-weight: 700" for _ in row.index]
                    return ["" for _ in row.index]

                # Index unsichtbar machen: minimale Breite und transparente Farbe
                sty = show_df.style.apply(_style_last, axis=1)
                # Modus: nicht umbrechen; Ergebnis: Zeilenumbrüche via "\n" darstellen
                try:
                    sty = sty.set_properties(subset=["Modus"], **{"white-space": "nowrap"})
                    sty = sty.set_properties(subset=["Ergebnis"], **{"white-space": "pre-line"})
                except Exception:
                    pass
                sty = sty.set_table_styles([
                    {"selector": "th.row_heading", "props": [
                        ("width","1px"),("min-width","1px"),("max-width","1px"),
                        ("padding","0"),("border","none"),("overflow","hidden"),
                        ("color","transparent")
                    ]},
                    {"selector": "tbody th", "props": [
                        ("width","1px"),("min-width","1px"),("max-width","1px"),
                        ("padding","0"),("border","none"),("overflow","hidden"),
                        ("color","transparent")
                    ]},
                    {"selector": "th.blank", "props": [
                        ("width","1px"),("min-width","1px"),("max-width","1px"),
                        ("padding","0"),("border","none"),("overflow","hidden"),
                        ("color","transparent")
                    ]},
                    {"selector": "th.col_heading", "props": [
                        ("white-space","nowrap"), ("text-align","left")
                    ]},
                ], overwrite=False)
                st.table(sty)
            else:
                st.info("Noch keine Spiele vorhanden.")
    # Spielen – neue UI für Spiele erstellen und verwalten
    with tabs[1]:
        st.subheader("Spielen")
        id_to_name, name_to_id = get_player_maps()
        me = st.session_state.get("player_id")

        # --- Erstellung: Modus per Tabs ---
        m_tabs = st.tabs(["Einzel", "Doppel", "Rundlauf"]) 

        # Einzel
        with m_tabs[0]:
            play_myself = st.checkbox("Ich spiele mit", value=True, help="Dein Name als Teilnehmer A. Deaktiviere, um ein Match für andere anzulegen.")
            if play_myself:
                opponent = st.selectbox("Gegner", [n for n in name_to_id.keys() if name_to_id[n] != me], key="einzel_opponent")
                c1, c2 = st.columns(2)
                s_a = c1.number_input("Deine Punkte", min_value=0, step=1, value=11, key="einzel_s_a")
                s_b = c2.number_input("Gegner Punkte", min_value=0, step=1, value=9, key="einzel_s_b")
                if st.button("✅", key="btn_send_single_me"):
                    create_pending_single(me, name_to_id[opponent], s_a, s_b)
                    clear_table_cache()
                    st.success("Einzel erstellt. Ein Teilnehmer muss bestätigen.")
                    st.rerun()
            else:
                a_player = st.selectbox("Spieler A", [n for n in name_to_id.keys()], key="einzel_a")
                b_player = st.selectbox("Spieler B", [n for n in name_to_id.keys() if n != a_player], key="einzel_b")
                c1, c2 = st.columns(2)
                s_a = c1.number_input("Punkte A", min_value=0, step=1, value=11, key="einzel2_s_a")
                s_b = c2.number_input("Punkte B", min_value=0, step=1, value=9, key="einzel2_s_b")
                if st.button("✅", key="btn_send_single_others"):
                    create_pending_single(me, name_to_id[b_player], s_a, s_b, a_id=name_to_id[a_player])
                    clear_table_cache()
                    st.success("Einzel erstellt. Ein Teilnehmer muss bestätigen.")
                    st.rerun()

        # Doppel
        with m_tabs[1]:
            play_myself_d = st.checkbox("Ich spiele mit", value=True, key="doppel_play_myself")
            if play_myself_d:
                partner = st.selectbox("Partner", [n for n in name_to_id.keys() if name_to_id[n] != me], key="d_partner")
                right1 = st.selectbox("Gegner 1", [n for n in name_to_id.keys() if name_to_id[n] not in (me, name_to_id[partner])], key="d_opp1")
                right2 = st.selectbox("Gegner 2", [n for n in name_to_id.keys() if name_to_id[n] not in (me, name_to_id[partner], name_to_id[right1])], key="d_opp2")
                c1, c2 = st.columns(2)
                s_a = c1.number_input("Eure Punkte", min_value=0, step=1, value=11, key="d_s_a")
                s_b = c2.number_input("Gegner Punkte", min_value=0, step=1, value=8, key="d_s_b")
                if st.button("✅", key="btn_send_double_me"):
                    # Direktes Insert mit fertigem Payload
                    payload = {
                        "datum": _utc_iso(pd.Timestamp.now(tz=TZ)),
                        "a1": me, "a2": name_to_id[partner], "b1": name_to_id[right1], "b2": name_to_id[right2],
                        "punktea": int(s_a), "punkteb": int(s_b),
                        "confa": False, "confb": False,
                    }
                    if table_has_creator("pending_doubles"):
                        payload["creator"] = me
                    sp.table("pending_doubles").insert(payload).execute()
                    clear_table_cache()
                    st.success("Doppel erstellt. Ein Teilnehmer muss bestätigen.")
                    st.rerun()
            else:
                a1 = st.selectbox("Spieler A1", [n for n in name_to_id.keys()], key="d_a1")
                a2 = st.selectbox("Spieler A2", [n for n in name_to_id.keys() if n != a1], key="d_a2")
                b1 = st.selectbox("Spieler B1", [n for n in name_to_id.keys() if n not in (a1, a2)], key="d_b1")
                b2 = st.selectbox("Spieler B2", [n for n in name_to_id.keys() if n not in (a1, a2, b1)], key="d_b2")
                c1, c2 = st.columns(2)
                s_a = c1.number_input("Punkte Team A", min_value=0, step=1, value=11, key="d2_s_a")
                s_b = c2.number_input("Punkte Team B", min_value=0, step=1, value=8, key="d2_s_b")
                if st.button("✅", key="btn_send_double_others"):
                    payload = {
                        "datum": _utc_iso(pd.Timestamp.now(tz=TZ)),
                        "a1": name_to_id[a1], "a2": name_to_id[a2], "b1": name_to_id[b1], "b2": name_to_id[b2],
                        "punktea": int(s_a), "punkteb": int(s_b),
                        "confa": False, "confb": False,
                    }
                    if table_has_creator("pending_doubles"):
                        payload["creator"] = me
                    sp.table("pending_doubles").insert(payload).execute()
                    clear_table_cache()
                    st.success("Doppel erstellt. Ein Teilnehmer muss bestätigen.")
                    st.rerun()

        # Rundlauf
        with m_tabs[2]:
            play_myself_r = st.checkbox("Ich spiele mit", value=True, key="round_play_myself")
            selectable = list(name_to_id.keys())
            default_sel = []
            if play_myself_r and st.session_state.get("player_name") in selectable:
                default_sel = [st.session_state.get("player_name")]
            selected = st.multiselect("Teilnehmer wählen", selectable, default=default_sel, help="Wähle alle Teilnehmer (inkl. dir selbst, falls du mitspielst).", key="round_multi")
            if len(selected) >= 2:
                c1, c2 = st.columns(2)
                # Eingabereihenfolge getauscht: erst 2., dann 1. (Sieger)
                second_name = c1.selectbox("Zweiter", selected, key="round_second")
                winner_candidate = [n for n in selected if n != second_name]
                winner_name = c2.selectbox("Sieger", winner_candidate, key="round_winner")
            else:
                winner_name, second_name = None, None
            if st.button("✅", key="btn_send_round"):
                pids = [name_to_id[n] for n in selected]
                if len(pids) < 2:
                    st.warning("Mindestens 2 Teilnehmer.")
                elif not winner_name or not second_name:
                    st.warning("Bitte Sieger und Zweiten wählen.")
                else:
                    create_pending_round(
                        me,
                        pids,
                        fin1_id=name_to_id[winner_name],
                        fin2_id=name_to_id[second_name],
                        winner_id=name_to_id[winner_name],
                    )
                    clear_table_cache()
                    st.success("Rundlauf erstellt (mit Sieger/Zweiter). Ein Teilnehmer muss bestätigen.")
                    st.rerun()

        st.divider()

        # --- Bestätigen (ich bin Teilnehmer, nicht Ersteller) ---
        col_head, col_btn = st.columns([8,1])  # heading wide, button stays tiny – fits better on mobile
        with col_head:
            st.markdown("### Offene Bestätigungen")
        with col_btn:
            if st.button("🔄", key="btn_refresh_confirmations"):
                clear_table_cache()
                st.rerun()
                # Refresh nur für die Pending-Listen
        pm = load_table("pending_matches")
        pdbl = load_table("pending_doubles")
        pr = load_table("pending_rounds")

        # --- Meine offenen Bestätigungen sammeln (pro Modus) ---
        info_rows_s, info_rows_d, info_rows_r = [], [], []
        # Einzel
        if not pm.empty:
            has_c = table_has_creator("pending_matches")
            if has_c:
                my_conf_s = pm[(pm["a"].astype(str).eq(str(me)) | pm["b"].astype(str).eq(str(me))) & (pm["creator"].astype(str) != str(me))]
            else:
                my_conf_s = pm[pm["b"].astype(str) == str(me)]
            for _, r in my_conf_s.iterrows():
                info_rows_s.append(r)
        # Doppel
        if not pdbl.empty:
            has_c_d = table_has_creator("pending_doubles")
            if has_c_d:
                part_mask = (pdbl[["a1","a2","b1","b2"]].astype(str) == str(me)).any(axis=1)
                my_conf_d = pdbl[part_mask & (pdbl["creator"].astype(str) != str(me))]
            else:
                my_conf_d = pdbl[(pdbl["a1"].astype(str) != str(me)) & ((pdbl["a2"].astype(str) == str(me)) | (pdbl["b1"].astype(str) == str(me)) | (pdbl["b2"].astype(str) == str(me)))]
            for _, r in my_conf_d.iterrows():
                info_rows_d.append(r)
        # Rundlauf
        if not pr.empty:
            has_c_r = table_has_creator("pending_rounds")
            if has_c_r:
                def _involved_not_creator(row):
                    teiln = [x for x in str(row.get("teilnehmer","")) .split(";") if x]
                    return (str(me) in teiln) and (str(row.get("creator")) != str(me))
                my_conf_r = pr[pr.apply(_involved_not_creator, axis=1)]
            else:
                def _is_involved_not_creator(row):
                    teiln = [x for x in str(row.get("teilnehmer","")) .split(";") if x]
                    return (str(me) in teiln) and (len(teiln) > 0 and teiln[0] != str(me))
                my_conf_r = pr[pr.apply(_is_involved_not_creator, axis=1)]
            for _, r in my_conf_r.iterrows():
                info_rows_r.append(r)

        # --- Global: Alle bestätigen (unter dem Refresh-Button) ---
        if any([info_rows_s, info_rows_d, info_rows_r]):
            if st.button("✅ Alle bestätigen", key="btn_accept_all_pending", type="primary"):
                try:
                    # Einzel
                    for r in info_rows_s:
                        confirm_pending_single(r)
                    # Doppel
                    for r in info_rows_d:
                        confirm_pending_double(r)
                    # Rundlauf
                    for r in info_rows_r:
                        confirm_pending_round(r)
                    clear_table_cache()
                    st.success("Alle bestätigbaren Spiele bestätigt.")
                    st.rerun()
                except Exception:
                    clear_table_cache()
                    st.warning("Massenbestätigung teilweise fehlgeschlagen. Seite neu laden und prüfen.")
        else:
            st.info("Keine offenen Bestätigungen.")

        # --- Karten-Ansicht der einzelnen Spiele (nur Ablehnen pro Karte) ---
        # Einzel-Karten
        me_name = user.get("name")
        for r in info_rows_s:
            render_single_vs_card(
                r, id_to_name,
                highlight_name=me_name,
                key=f"trej_s_{r['id']}",
                on_reject=lambda rid: (reject_pending("pending_matches", rid), clear_table_cache(), st.rerun()),
                button_label="❌ Ablehnen",
            )

        # Doppel-Karten
        me_name = user.get("name")
        for r in info_rows_d:
            render_double_vs_card(
                r, id_to_name,
                highlight_name=me_name,
                key=f"trej_d_{r['id']}",
                on_reject=lambda rid: (reject_pending("pending_doubles", rid), clear_table_cache(), st.rerun()),
                button_label="❌ Ablehnen",
            )

        # Rundlauf-Karten
        me_name = user.get("name")
        for r in info_rows_r:
            render_round_vs_card(
                r, id_to_name,
                highlight_name=me_name,
                key=f"trej_r_{r['id']}",
                on_reject=lambda rid: (reject_pending("pending_rounds", rid), clear_table_cache(), st.rerun()),
                button_label="❌ Ablehnen",
            )

        # --- Von mir erstellt (ich kann abbrechen) ---
        st.markdown("### Ausstehende Bestätigungen")
        if not pm.empty:
            if table_has_creator("pending_matches"):
                mine = pm[pm["creator"].astype(str) == str(me)]
            else:
                mine = pm[pm["a"].astype(str) == str(me)]
            me_name = user.get("name")
            for _, r in mine.iterrows():
                render_single_vs_card(
                    r, id_to_name,
                    highlight_name=me_name,
                    key=f"cancel_s_{r['id']}",
                    on_reject=lambda rid: (reject_pending("pending_matches", rid), clear_table_cache(), st.rerun()),
                    button_label="❌ Ablehnen",
                )

        if not pdbl.empty:
            if table_has_creator("pending_doubles"):
                mine = pdbl[pdbl["creator"].astype(str) == str(me)]
            else:
                mine = pdbl[pdbl["a1"].astype(str) == str(me)]
            me_name = user.get("name")
            for _, r in mine.iterrows():
                render_double_vs_card(
                    r, id_to_name,
                    highlight_name=me_name,
                    key=f"cancel_d_{r['id']}",
                    on_reject=lambda rid: (reject_pending("pending_doubles", rid), clear_table_cache(), st.rerun()),
                    button_label="❌ Ablehnen",
                )

        if not pr.empty:
            if table_has_creator("pending_rounds"):
                mine = pr[pr["creator"].astype(str) == str(me)]
            else:
                def _created_by_me(row):
                    teiln = [x for x in str(row.get("teilnehmer","")) .split(";") if x]
                    return len(teiln) > 0 and teiln[0] == str(me)
                mine = pr[pr.apply(_created_by_me, axis=1)]
            me_name = user.get("name")
            for _, r in mine.iterrows():
                render_round_vs_card(
                    r, id_to_name,
                    highlight_name=me_name,
                    key=f"cancel_r_{r['id']}",
                    on_reject=lambda rid: (reject_pending("pending_rounds", rid), clear_table_cache(), st.rerun()),
                    button_label="❌ Ablehnen",
                )

    # Account – mit Logout‑Button
    with tabs[2]:
        st.subheader("Account")
        st.write(f"Angemeldet als **{user.get('name','Unbekannt')}**")

        # --- Profil: Name ändern ---
        st.markdown("### Profil")
        with st.form("form_change_name"):
            new_name = st.text_input("Neuer Anzeigename", value=user.get("name", ""))
            save_name = st.form_submit_button("Name speichern", type="primary")
        if save_name:
            nn = (new_name or "").strip()
            if not nn:
                st.warning("Bitte einen gültigen Namen eingeben.")
            else:
                # Duplikate ignorieren Groß/Kleinschreibung & Leerzeichen
                try:
                    rows = sp.table("players").select("id,name").execute().data or []
                except Exception:
                    rows = []
                def _norm(s: str) -> str:
                    return "".join(str(s or "").split()).lower()
                me_id = str(user.get("id"))
                if any(_norm(r.get("name")) == _norm(nn) and str(r.get("id")) != me_id for r in rows):
                    st.error("Name bereits vergeben (Groß-/Kleinschreibung & Leerzeichen ignoriert).")
                else:
                    try:
                        sp.table("players").update({"name": nn}).eq("id", me_id).execute()
                        # Caches leeren & Session aktualisieren
                        try:
                            load_table.clear()
                        except Exception:
                            pass
                        try:
                            get_player_maps.clear()
                        except Exception:
                            pass
                        st.session_state.player_name = nn
                        st.success("Name aktualisiert.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Konnte Namen nicht speichern: {e}")

        # --- Sicherheit: PIN ändern ---
        st.markdown("### Sicherheit")
        with st.form("form_change_pin"):
            old_pin = st.text_input("Aktuelle PIN", type="password")
            new_pin1 = st.text_input("Neue PIN (4-stellig)", type="password")
            new_pin2 = st.text_input("Neue PIN bestätigen", type="password")
            save_pin = st.form_submit_button("PIN speichern", type="primary")
        if save_pin:
            cur_stored = str(user.get("pin", ""))
            if not old_pin or not new_pin1 or not new_pin2:
                st.warning("Bitte alle Felder ausfüllen.")
            elif not check_pin(old_pin, cur_stored):
                st.error("Aktuelle PIN ist falsch.")
            elif new_pin1 != new_pin2:
                st.error("Neue PINs stimmen nicht überein.")
            elif len(new_pin1) != 4 or not new_pin1.isdigit():
                st.warning("Die neue PIN muss 4 Ziffern haben.")
            else:
                try:
                    hp = hash_pin(new_pin1)
                    sp.table("players").update({"pin": hp}).eq("id", user.get("id")).execute()
                    st.success("PIN aktualisiert.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Konnte PIN nicht speichern: {e}")

        # Logout bleibt am Ende
        if st.button("Logout", key="btn_logout_account", type="primary"):
            try:
                sp.table("players").update({"auto_token": None}).eq("id", user.get("id")).execute()
            except Exception:
                pass
            for k in ("logged_in","player_id","player_name"):
                st.session_state.pop(k, None)
            st.query_params.clear()
            st.rerun()

# endregion

# region auth_section
# Login/Registrierung als eigenes Fenster. Wenn nicht eingeloggt, zeigen wir NUR dieses Fenster
# und stoppen danach das Rendering der restlichen Seite.
if not st.session_state.get("logged_in"):
    # Versuche, die gesamte Login/Registrierung in einen echten Streamlit-Container mit Rahmen zu legen
    try:
        with st.container(border=True):
            st.markdown('<div class="auth-title">🔐 Login & Registrierung</div>', unsafe_allow_html=True)
            login_register_ui()
    except TypeError:
        # Fallback für ältere Streamlit-Versionen ohne `border=True`
        st.markdown(
            """
            <style>
            .auth-card { max-width: 520px; margin: 6vh auto; padding: 1.0rem 1.0rem 0.75rem; border: 1px solid rgba(255,255,255,0.20); border-radius: 12px; background: rgba(0,0,0,0.25); box-shadow: 0 8px 28px rgba(0,0,0,.35); }
            .auth-title { text-align:center; font-size: 1.1rem; margin: 0.25rem 0 0.75rem; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="auth-card"><div class="auth-title">🔐 Login & Registrierung</div>', unsafe_allow_html=True)
        login_register_ui()
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()
else:
    logged_in_ui()
# endregion
