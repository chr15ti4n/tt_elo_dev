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

# endregion

# region game_helpers
from datetime import datetime

def _utc_iso(ts) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").isoformat()

def create_pending_single(creator_id: str, opponent_id: str, s_a: int, s_b: int):
    payload = {
        "datum": _utc_iso(pd.Timestamp.now(tz=TZ)),
        "a": creator_id, "b": opponent_id,
        "punktea": int(s_a), "punkteb": int(s_b),
        "confa": True, "confb": False,
    }
    sp.table("pending_matches").insert(payload).execute()

def create_pending_double(creator_id: str, partner_id: str, opp1_id: str, opp2_id: str, s_a: int, s_b: int):
    payload = {
        "datum": _utc_iso(pd.Timestamp.now(tz=TZ)),
        "a1": creator_id, "a2": partner_id,
        "b1": opp1_id, "b2": opp2_id,
        "punktea": int(s_a), "punkteb": int(s_b),
        "confa": True, "confb": False,
    }
    sp.table("pending_doubles").insert(payload).execute()

def create_pending_round(creator_id: str, participant_ids: list[str]):
    teilnehmer = ";".join([creator_id] + [pid for pid in participant_ids if pid != creator_id])
    payload = {
        "datum": _utc_iso(pd.Timestamp.now(tz=TZ)),
        "teilnehmer": teilnehmer,
        "finalisten": None,
        "sieger": None,
        "confa": True, "confb": False,
    }
    sp.table("pending_rounds").insert(payload).execute()

# --- Confirm / Reject (ohne ELO-Update; das bauen wir im nächsten Schritt) ---

def confirm_pending_single(row: pd.Series):
    sp.table("matches").insert({
        "datum": _utc_iso(pd.Timestamp(row["datum"])),
        "a": row["a"], "b": row["b"],
        "punktea": int(row["punktea"]), "punkteb": int(row["punkteb"]),
    }).execute()
    sp.table("pending_matches").delete().eq("id", row["id"]).execute()


def confirm_pending_double(row: pd.Series):
    sp.table("doubles").insert({
        "datum": _utc_iso(pd.Timestamp(row["datum"])),
        "a1": row["a1"], "a2": row["a2"],
        "b1": row["b1"], "b2": row["b2"],
        "punktea": int(row["punktea"]), "punkteb": int(row["punkteb"]),
    }).execute()
    sp.table("pending_doubles").delete().eq("id", row["id"]).execute()


def confirm_pending_round(row: pd.Series):
    sp.table("rounds").insert({
        "datum": _utc_iso(pd.Timestamp(row["datum"])),
        "teilnehmer": row["teilnehmer"],
        "finalisten": row.get("finalisten"),
        "sieger": row.get("sieger"),
    }).execute()
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

    # Übersicht
    with tabs[0]:
        logged_in_header(user)
        st.markdown("""
        **Übersicht**
        
        Hier bauen wir als Nächstes die Inhalte auf (z. B. letzte Spiele, Win‑Streak, offene Bestätigungen).
        """)

    # Spielen – neue UI für Spiele erstellen und verwalten
    with tabs[1]:
        st.subheader("Spielen")
        id_to_name, name_to_id = get_player_maps()
        me = st.session_state.get("player_id")

        # --- Erstellung ---
        mode = st.radio("Modus wählen", ["Einzel", "Doppel", "Rundlauf"], horizontal=True)
        if mode == "Einzel":
            opponent = st.selectbox("Gegner", [n for n in name_to_id.keys() if name_to_id[n] != me])
            c1, c2 = st.columns(2)
            s_a = c1.number_input("Deine Punkte", min_value=0, step=1, value=11)
            s_b = c2.number_input("Gegner Punkte", min_value=0, step=1, value=9)
            if st.button("Einladung senden (Einzel)"):
                create_pending_single(me, name_to_id[opponent], s_a, s_b)
                clear_table_cache()
                st.success("Einladung erstellt. Der Gegner muss bestätigen oder ablehnen.")
                st.rerun()

        elif mode == "Doppel":
            partner = st.selectbox("Partner", [n for n in name_to_id.keys() if name_to_id[n] != me])
            right1 = st.selectbox("Gegner 1", [n for n in name_to_id.keys() if name_to_id[n] not in (me, name_to_id[partner])])
            right2 = st.selectbox("Gegner 2", [n for n in name_to_id.keys() if name_to_id[n] not in (me, name_to_id[partner], name_to_id[right1])])
            c1, c2 = st.columns(2)
            s_a = c1.number_input("Eure Punkte", min_value=0, step=1, value=11)
            s_b = c2.number_input("Gegner Punkte", min_value=0, step=1, value=8)
            if st.button("Einladung senden (Doppel)"):
                create_pending_double(me, name_to_id[partner], name_to_id[right1], name_to_id[right2], s_a, s_b)
                clear_table_cache()
                st.success("Doppel-Einladung erstellt.")
                st.rerun()

        else:  # Rundlauf
            teilnehmer = st.multiselect("Teilnehmer wählen", [n for n in name_to_id.keys() if name_to_id[n] != me])
            if st.button("Rundlauf eintragen (pending)"):
                if len(teilnehmer) < 1:
                    st.warning("Mindestens 1 weiterer Teilnehmer.")
                else:
                    others = [name_to_id[n] for n in teilnehmer]
                    create_pending_round(me, others)
                    clear_table_cache()
                    st.success("Rundlauf gespeichert (pending).")
                    st.rerun()

        st.divider()

        # --- Bestätigen (ich bin Teilnehmer, nicht Ersteller) ---
        st.markdown("### Offene Bestätigungen")
        pm = load_table("pending_matches")
        pdbl = load_table("pending_doubles")
        pr = load_table("pending_rounds")

        if not pm.empty:
            my_conf = pm[(pm["b"].astype(str) == str(me)) & (~pm["confa"] | ~pm["confb"])]
            for _, r in my_conf.iterrows():
                a_n = id_to_name.get(str(r["a"]), r["a"])
                b_n = id_to_name.get(str(r["b"]), r["b"])
                line = f"Einzel: {a_n} vs {b_n}  {int(r['punktea'])}:{int(r['punkteb'])}"
                c1, c2 = st.columns([3,1])
                c1.write(line)
                if c2.button("Bestätigen", key=f"conf_s_{r['id']}"):
                    confirm_pending_single(r)
                    clear_table_cache()
                    st.success("Einzel bestätigt.")
                    st.rerun()
                if c2.button("Ablehnen", key=f"rej_s_{r['id']}"):
                    reject_pending("pending_matches", r["id"])
                    clear_table_cache()
                    st.info("Einzel abgelehnt.")
                    st.rerun()

        if not pdbl.empty:
            mask = (
                (pdbl["a1"].astype(str) != str(me)) & (
                    (pdbl["a2"].astype(str) == str(me)) |
                    (pdbl["b1"].astype(str) == str(me)) |
                    (pdbl["b2"].astype(str) == str(me))
                ) & (~pdbl["confa"] | ~pdbl["confb"]))
            my_conf = pdbl[mask]
            for _, r in my_conf.iterrows():
                a1 = id_to_name.get(str(r["a1"]), r["a1"]); a2 = id_to_name.get(str(r["a2"]), r["a2"])
                b1 = id_to_name.get(str(r["b1"]), r["b1"]); b2 = id_to_name.get(str(r["b2"]), r["b2"])
                line = f"Doppel: {a1}/{a2} vs {b1}/{b2}  {int(r['punktea'])}:{int(r['punkteb'])}"
                c1, c2 = st.columns([3,1])
                c1.write(line)
                if c2.button("Bestätigen", key=f"conf_d_{r['id']}"):
                    confirm_pending_double(r)
                    clear_table_cache()
                    st.success("Doppel bestätigt.")
                    st.rerun()
                if c2.button("Ablehnen", key=f"rej_d_{r['id']}"):
                    reject_pending("pending_doubles", r["id"])
                    clear_table_cache()
                    st.info("Doppel abgelehnt.")
                    st.rerun()

        if not pr.empty:
            # bestätigbar, wenn ich Teilnehmer bin und nicht der Ersteller (Ersteller = erster Eintrag in teilnehmer)
            def _is_involved_not_creator(row):
                teiln = [x for x in str(row.get("teilnehmer","")) .split(";") if x]
                return (str(me) in teiln) and (len(teiln) > 0 and teiln[0] != str(me))
            my_conf = pr[pr.apply(_is_involved_not_creator, axis=1)]
            for _, r in my_conf.iterrows():
                teiln = [id_to_name.get(pid, pid) for pid in str(r["teilnehmer"]).split(";") if pid]
                line = f"Rundlauf: {', '.join(teiln)}"
                c1, c2 = st.columns([3,1])
                c1.write(line)
                if c2.button("Bestätigen", key=f"conf_r_{r['id']}"):
                    confirm_pending_round(r)
                    clear_table_cache()
                    st.success("Rundlauf bestätigt.")
                    st.rerun()
                if c2.button("Ablehnen", key=f"rej_r_{r['id']}"):
                    reject_pending("pending_rounds", r["id"])
                    clear_table_cache()
                    st.info("Rundlauf abgelehnt.")
                    st.rerun()

        st.divider()

        # --- Von mir erstellt (ich kann abbrechen) ---
        st.markdown("### Von dir erstellt")
        if not pm.empty:
            mine = pm[pm["a"].astype(str) == str(me)]
            for _, r in mine.iterrows():
                a_n = id_to_name.get(str(r["a"]), r["a"]) ; b_n = id_to_name.get(str(r["b"]), r["b"]) 
                line = f"Einzel: {a_n} vs {b_n}  {int(r['punktea'])}:{int(r['punkteb'])}"
                c1, c2 = st.columns([3,1])
                c1.write(line)
                if c2.button("Abbrechen", key=f"cancel_s_{r['id']}"):
                    reject_pending("pending_matches", r["id"])
                    clear_table_cache()
                    st.info("Einladung verworfen.")
                    st.rerun()

        if not pdbl.empty:
            mine = pdbl[pdbl["a1"].astype(str) == str(me)]
            for _, r in mine.iterrows():
                a1 = id_to_name.get(str(r["a1"]), r["a1"]); a2 = id_to_name.get(str(r["a2"]), r["a2"])
                b1 = id_to_name.get(str(r["b1"]), r["b1"]); b2 = id_to_name.get(str(r["b2"]), r["b2"])
                line = f"Doppel: {a1}/{a2} vs {b1}/{b2}  {int(r['punktea'])}:{int(r['punkteb'])}"
                c1, c2 = st.columns([3,1])
                c1.write(line)
                if c2.button("Abbrechen", key=f"cancel_d_{r['id']}"):
                    reject_pending("pending_doubles", r["id"])
                    clear_table_cache()
                    st.info("Einladung verworfen.")
                    st.rerun()

        if not pr.empty:
            def _created_by_me(row):
                teiln = [x for x in str(row.get("teilnehmer","")) .split(";") if x]
                return len(teiln) > 0 and teiln[0] == str(me)
            mine = pr[pr.apply(_created_by_me, axis=1)]
            for _, r in mine.iterrows():
                teiln = [id_to_name.get(pid, pid) for pid in str(r["teilnehmer"]).split(";") if pid]
                line = f"Rundlauf: {', '.join(teiln)}"
                c1, c2 = st.columns([3,1])
                c1.write(line)
                if c2.button("Abbrechen", key=f"cancel_r_{r['id']}"):
                    reject_pending("pending_rounds", r["id"])
                    clear_table_cache()
                    st.info("Rundlauf verworfen.")
                    st.rerun()

    # Account – mit Logout‑Button
    with tabs[2]:
        st.subheader("Account")
        st.write(f"Angemeldet als **{user.get('name','Unbekannt')}**")
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
    st.markdown(
        """
        <style>
        .auth-card { max-width: 520px; margin: 6vh auto; padding: 1.25rem 1.25rem 0.5rem; border: 1px solid rgba(255,255,255,0.15); border-radius: 12px; background: rgba(0,0,0,0.25); box-shadow: 0 8px 28px rgba(0,0,0,.35); }
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
