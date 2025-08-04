

# tt-elo – Streamlit + Supabase (PIN-basiert)
# -------------------------------------------------
# Voraussetzungen:
#   pip install streamlit supabase bcrypt pandas numpy pytz
# Streamlit Secrets (.streamlit/secrets.toml):
#   [supabase]
#   url = "https://<YOUR-PROJECT>.supabase.co"
#   key = "<YOUR-ANON-OR-SERVICE-KEY>"
# -------------------------------------------------

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
import bcrypt
from typing import Dict, List, Optional, Tuple
from supabase import create_client, Client

# ---- App Setup ----
st.set_page_config(page_title="tt-elo", page_icon="🏓", layout="wide")
TZ = ZoneInfo("Europe/Berlin")
ADMINS: List[str] = ["Chris"]  # Admin-Benutzername für Full-Rebuild o.ä.

# ---- Supabase ----
@st.cache_resource
def get_supabase() -> Client:
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase = get_supabase()

# ---- PIN-Hashing ----
def hash_pin(pin: str) -> str:
    return bcrypt.hashpw(pin.encode(), bcrypt.gensalt()).decode()

def check_pin(entered: str, stored: str) -> bool:
    """Vergleicht eingegebene PIN mit gespeichertem Wert (unterstützt Legacy-Klartext)."""
    if stored.startswith("$2b$") or stored.startswith("$2a$"):
        try:
            return bcrypt.checkpw(entered.encode(), stored.encode())
        except Exception:
            return False
    # Legacy: Klartextgleichheit
    return entered == stored

# ---- Cache-Layer für Tabellen ----
@st.cache_data(ttl=30)
def load_table(table_name: str) -> pd.DataFrame:
    """Lädt eine Supabase-Tabelle als DataFrame. Spalten klein, Datum -> TZ."""
    res = supabase.table(table_name).select("*").execute()
    data = res.data or []
    df = pd.DataFrame(data)
    if df.empty:
        # minimale Schemas (kompatibel zum Alt-Code)
        schemas = {
            "players": [
                "id","name","pin","elo","siege","niederlagen","spiele",
                "d_elo","d_siege","d_niederlagen","d_spiele",
                "r_elo","r_siege","r_zweite","r_niederlagen","r_spiele",
                "g_elo","created_at","updated_at"
            ],
            "matches": ["id","datum","a","b","punktea","punkteb","created_at"],
            "pending_matches": ["id","datum","a","b","punktea","punkteb","confa","confb","created_at"],
            "doubles": ["id","datum","a1","a2","b1","b2","punktea","punkteb","created_at"],
            "pending_doubles": ["id","datum","a1","a2","b1","b2","punktea","punkteb","confa","confb","created_at"],
            "rounds": ["id","datum","teilnehmer","finalisten","sieger","created_at"],
            "pending_rounds": ["id","datum","teilnehmer","finalisten","sieger","confa","confb","created_at"],
        }
        df = pd.DataFrame(columns=schemas.get(table_name, []))
    # Normalisiere Spaltennamen
    df.columns = [str(c).lower() for c in df.columns]
    # Datum -> TZ
    if "datum" in df.columns and not df.empty:
        df["datum"] = pd.to_datetime(df["datum"], utc=True, errors="coerce").dt.tz_convert(TZ)
    return df

def clear_table_cache():
    load_table.clear()

# ---- Hilfen: Player-Maps ----
@st.cache_data(ttl=30)
def get_player_maps() -> Tuple[Dict[str,str], Dict[str,str]]:
    players = load_table("players")
    id_to_name = {}
    name_to_id = {}
    for _, r in players.iterrows():
        pid = str(r.get("id"))
        nm = str(r.get("name"))
        id_to_name[pid] = nm
        name_to_id[nm] = pid
    return id_to_name, name_to_id

# ---- ELO ----
def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

def update_elo(r_a: float, r_b: float, score_a: float, k: float = 32.0) -> Tuple[int, int]:
    ea = expected_score(r_a, r_b)
    eb = 1.0 - ea
    new_a = round(r_a + k * (score_a - ea))
    new_b = round(r_b + k * ((1.0 - score_a) - eb))
    return int(new_a), int(new_b)

def compute_gelo(elo: int, d_elo: int, r_elo: int) -> int:
    # Gewichtung wie im Alt-Code: 0.6 / 0.25 / 0.15
    return int(round(0.6 * int(elo) + 0.25 * int(d_elo) + 0.15 * int(r_elo)))

# ---- Auth (PIN) ----
def login_ui():
    st.subheader("Login / Registrierung")
    tab_login, tab_register = st.tabs(["Einloggen", "Registrieren"])

    with tab_login:
        name = st.text_input("Spielername", key="login_name")
        pin  = st.text_input("PIN", type="password", key="login_pin")
        if st.button("Einloggen"):
            players = load_table("players")
            if name not in set(players["name"].astype(str)):
                st.error("Spielername nicht gefunden.")
                return
            rec = players.loc[players["name"] == name].iloc[0]
            stored_pin = str(rec.get("pin", ""))
            if not check_pin(pin, stored_pin):
                st.error("PIN falsch.")
                return
            # Falls Klartext -> upgraden auf bcrypt
            if not (stored_pin.startswith("$2b$") or stored_pin.startswith("$2a$")):
                new_hash = hash_pin(pin)
                supabase.table("players").update({"pin": new_hash}).eq("id", rec["id"]).execute()
                clear_table_cache()
            st.session_state.logged_in = True
            st.session_state.player_id = rec["id"]
            st.session_state.player_name = rec["name"]
            st.success(f"Willkommen, {rec['name']}!")
            st.rerun()

    with tab_register:
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
            players = load_table("players")
            if r_name in set(players["name"].astype(str)):
                st.warning("Spieler existiert bereits.")
                return
            payload = {
                "name": r_name,
                "pin": hash_pin(r_pin1),
                "elo": 1200, "siege": 0, "niederlagen": 0, "spiele": 0,
                "d_elo": 1200, "d_siege": 0, "d_niederlagen": 0, "d_spiele": 0,
                "r_elo": 1200, "r_siege": 0, "r_zweite": 0, "r_niederlagen": 0, "r_spiele": 0,
                "g_elo": 1200,
            }
            supabase.table("players").insert(payload).execute()
            clear_table_cache()
            st.success("Registriert! Bitte einloggen.")

# ---- UI-Komponenten ----
def header(user: Optional[dict]):
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:12px;">
          <div style="font-size:28px;">🏓 <b>tt-elo</b></div>
          <div style="opacity:0.7;">Streamlit + Supabase</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if user:
        cols = st.columns(4)
        cols[0].metric("G-ELO", int(user.get("g_elo", 1200)))
        cols[1].metric("Einzel", int(user.get("elo", 1200)))
        cols[2].metric("Doppel", int(user.get("d_elo", 1200)))
        cols[3].metric("Rundlauf", int(user.get("r_elo", 1200)))

# ---- Helfer zum Schreiben ----
def upsert_player_stats(pid: str, payload: dict):
    # Rechne G-ELO, falls Einzel/Doppel/Rundlauf verändert wurden
    if any(k in payload for k in ("elo", "d_elo", "r_elo")):
        # hole alte Werte zum Zusammensetzen
        rec = supabase.table("players").select("elo,d_elo,r_elo").eq("id", pid).single().execute().data or {}
        elo = payload.get("elo", rec.get("elo", 1200))
        d_elo = payload.get("d_elo", rec.get("d_elo", 1200))
        r_elo = payload.get("r_elo", rec.get("r_elo", 1200))
        payload["g_elo"] = compute_gelo(int(elo), int(d_elo), int(r_elo))
    supabase.table("players").update(payload).eq("id", pid).execute()

# ---- Bestätigungslogik (inkrementell) ----
def confirm_single(row: pd.Series):
    # 1) Persistiere Match
    supabase.table("matches").insert({
        "datum": pd.Timestamp(row["datum"]).tz_convert("UTC").isoformat(),
        "a": row["a"], "b": row["b"],
        "punktea": int(row["punktea"]), "punkteb": int(row["punkteb"])
    }).execute()
    # 2) Update Spieler-ELO & Zähler
    a = supabase.table("players").select("id,elo,siege,niederlagen,spiele,d_elo,r_elo").eq("id", row["a"]).single().execute().data
    b = supabase.table("players").select("id,elo,siege,niederlagen,spiele,d_elo,r_elo").eq("id", row["b"]).single().execute().data
    if not a or not b:
        return
    score_a = 1.0 if row["punktea"] > row["punkteb"] else 0.0
    new_a, new_b = update_elo(int(a.get("elo",1200)), int(b.get("elo",1200)), score_a, k=32)
    a_payload = {
        "elo": int(new_a),
        "siege": int(a.get("siege",0)) + (1 if score_a==1 else 0),
        "niederlagen": int(a.get("niederlagen",0)) + (0 if score_a==1 else 1),
        "spiele": int(a.get("spiele",0)) + 1,
    }
    b_payload = {
        "elo": int(new_b),
        "siege": int(b.get("siege",0)) + (0 if score_a==1 else 1),
        "niederlagen": int(b.get("niederlagen",0)) + (1 if score_a==1 else 0),
        "spiele": int(b.get("spiele",0)) + 1,
    }
    upsert_player_stats(a["id"], a_payload)
    upsert_player_stats(b["id"], b_payload)


def confirm_double(row: pd.Series):
    # 1) Persistiere Doppel
    supabase.table("doubles").insert({
        "datum": pd.Timestamp(row["datum"]).tz_convert("UTC").isoformat(),
        "a1": row["a1"], "a2": row["a2"], "b1": row["b1"], "b2": row["b2"],
        "punktea": int(row["punktea"]), "punkteb": int(row["punkteb"])
    }).execute()
    # 2) Team-Rating = Durchschnitt der Partner
    def get_two(p1: str, p2: str):
        r1 = supabase.table("players").select("id,d_elo,elo,r_elo").eq("id", p1).single().execute().data
        r2 = supabase.table("players").select("id,d_elo,elo,r_elo").eq("id", p2).single().execute().data
        return r1, r2
    A1, A2 = get_two(row["a1"], row["a2"])
    B1, B2 = get_two(row["b1"], row["b2"])    
    if not A1 or not A2 or not B1 or not B2:
        return
    r_a = (int(A1.get("d_elo",1200)) + int(A2.get("d_elo",1200))) / 2
    r_b = (int(B1.get("d_elo",1200)) + int(B2.get("d_elo",1200))) / 2
    score_a = 1.0 if row["punktea"] > row["punkteb"] else 0.0
    new_a, new_b = update_elo(r_a, r_b, score_a, k=24)  # etwas konservativer für Doppel

    # delta vs. altes Team-Rating gleichmäßig verteilen
    delta_a = int(round(new_a - r_a))
    delta_b = int(round(new_b - r_b))
    for rec, d in [(A1, delta_a), (A2, delta_a), (B1, delta_b), (B2, delta_b)]:
        payload = {
            "d_elo": int(rec.get("d_elo",1200)) + d,
            "d_spiele": int(rec.get("d_spiele",0)) + 1,
            "d_siege": int(rec.get("d_siege",0)) + (1 if (score_a==1 and rec in (A1,A2)) or (score_a==0 and rec in (B1,B2)) else 0),
            "d_niederlagen": int(rec.get("d_niederlagen",0)) + (1 if (score_a==0 and rec in (A1,A2)) or (score_a==1 and rec in (B1,B2)) else 0),
        }
        upsert_player_stats(rec["id"], payload)


def confirm_round(row: pd.Series):
    # 1) Persistiere Rundlauf
    supabase.table("rounds").insert({
        "datum": pd.Timestamp(row["datum"]).tz_convert("UTC").isoformat(),
        "teilnehmer": row["teilnehmer"],
        "finalisten": row.get("finalisten"),
        "sieger": row.get("sieger"),
    }).execute()

    # 2) Inkrementelle ELO/Zähler – einfache Regel:
    teilnehmer = [pid for pid in str(row["teilnehmer"]).split(";") if pid]
    winner = str(row.get("sieger")) if row.get("sieger") else None
    finalists = [pid for pid in str(row.get("finalisten") or "").split(";") if pid]

    # Durchschnitt als Bezug
    current = {}
    for pid in teilnehmer:
        rec = supabase.table("players").select("id,r_elo,r_siege,r_zweite,r_niederlagen,r_spiele,elo,d_elo").eq("id", pid).single().execute().data
        if rec:
            current[pid] = rec
    if not current:
        return
    avg = np.mean([int(r.get("r_elo",1200)) for r in current.values()])

    # Gewinner +10 vs. Offset, alle anderen - (10/ (n-1)) sodass Summe ~0
    n = len(current)
    gain = 10
    for pid, rec in current.items():
        if winner and pid == winner:
            new_r = int(rec.get("r_elo",1200)) + gain
            payload = {
                "r_elo": new_r,
                "r_siege": int(rec.get("r_siege",0)) + 1,
                "r_spiele": int(rec.get("r_spiele",0)) + 1,
            }
        else:
            lose = round(gain / max(1, n-1))
            new_r = int(rec.get("r_elo",1200)) - lose
            payload = {
                "r_elo": new_r,
                "r_niederlagen": int(rec.get("r_niederlagen",0)) + 1,
                "r_spiele": int(rec.get("r_spiele",0)) + 1,
            }
        if finalists and pid in finalists and (not winner or pid != winner):
            payload["r_zweite"] = int(rec.get("r_zweite",0)) + 1
        upsert_player_stats(pid, payload)

# ---- Eintragungs-UI ----
def create_invite_ui(player_id: str):
    id_to_name, name_to_id = get_player_maps()
    st.subheader("Match eintragen")
    mode = st.radio("Modus", ["Einzel", "Doppel", "Rundlauf"], horizontal=True)

    if mode == "Einzel":
        opponent = st.selectbox("Gegner", [n for n in name_to_id.keys() if name_to_id[n] != player_id])
        col1, col2 = st.columns(2)
        s_a = col1.number_input("Punkte A (du)", min_value=0, step=1, value=11)
        s_b = col2.number_input("Punkte B", min_value=0, step=1, value=9)
        if st.button("Einladung senden"):
            payload = {
                "datum": datetime.now(TZ).astimezone(ZoneInfo("UTC")).isoformat(),
                "a": player_id, "b": name_to_id[opponent],
                "punktea": int(s_a), "punkteb": int(s_b),
                "confa": True, "confb": False,
            }
            supabase.table("pending_matches").insert(payload).execute()
            clear_table_cache()
            st.success("Einzel-Einladung erstellt.")

    elif mode == "Doppel":
        left = st.selectbox("Partner", [n for n in name_to_id.keys() if name_to_id[n] != player_id])
        right1 = st.selectbox("Gegner 1", [n for n in name_to_id.keys() if n not in (left,) and name_to_id[n] != player_id])
        right2 = st.selectbox("Gegner 2", [n for n in name_to_id.keys() if n not in (left, right1) and name_to_id[n] != player_id])
        col1, col2 = st.columns(2)
        s_a = col1.number_input("Punkte A (ihr)", min_value=0, step=1, value=11)
        s_b = col2.number_input("Punkte B", min_value=0, step=1, value=8)
        if st.button("Einladung senden"):
            payload = {
                "datum": datetime.now(TZ).astimezone(ZoneInfo("UTC")).isoformat(),
                "a1": player_id, "a2": name_to_id[left],
                "b1": name_to_id[right1], "b2": name_to_id[right2],
                "punktea": int(s_a), "punkteb": int(s_b),
                "confa": True, "confb": False,
            }
            supabase.table("pending_doubles").insert(payload).execute()
            clear_table_cache()
            st.success("Doppel-Einladung erstellt.")

    else:  # Rundlauf
        teilnehmer = st.multiselect("Teilnehmer", list(name_to_id.keys()))
        finalists = st.multiselect("Finalisten (optional)", teilnehmer)
        sieger = st.selectbox("Sieger (optional)", [""] + teilnehmer)
        if st.button("Rundlauf eintragen"):
            if len(teilnehmer) < 2:
                st.warning("Mindestens 2 Teilnehmer.")
            else:
                payload = {
                    "datum": datetime.now(TZ).astimezone(ZoneInfo("UTC")).isoformat(),
                    "teilnehmer": ";".join([name_to_id[n] for n in teilnehmer]),
                    "finalisten": ";".join([name_to_id[n] for n in finalists]) if finalists else None,
                    "sieger": name_to_id[sieger] if sieger else None,
                    "confa": True, "confb": False,
                }
                supabase.table("pending_rounds").insert(payload).execute()
                clear_table_cache()
                st.success("Rundlauf gespeichert (pending).")

# ---- Bestätigungs-UI ----
def confirm_ui(player_id: str):
    id_to_name, _ = get_player_maps()
    st.subheader("Bestätigungen")

    # Einzel
    pm = load_table("pending_matches")
    if not pm.empty:
        mine = pm[(pm["a"] == player_id) | (pm["b"] == player_id)].copy()
        if not mine.empty:
            st.markdown("**Einzel**")
            for _, r in mine.iterrows():
                a_n = id_to_name.get(r["a"], r["a"])
                b_n = id_to_name.get(r["b"], r["b"])
                cols = st.columns([3,1,1])
                cols[0].write(f"{a_n} vs {b_n}  {int(r['punktea'])}:{int(r['punkteb'])}")
                if cols[1].button("✅", key=f"conf_s_{r['id']}"):
                    confirm_single(r)
                    supabase.table("pending_matches").delete().eq("id", r["id"]).execute()
                    clear_table_cache()
                    st.success("Einzel bestätigt.")
                    st.experimental_rerun()
                if cols[2].button("❌", key=f"rej_s_{r['id']}"):
                    supabase.table("pending_matches").delete().eq("id", r["id"]).execute()
                    clear_table_cache()
                    st.info("Einzel verworfen.")
                    st.experimental_rerun()

    # Doppel
    pd_ = load_table("pending_doubles")
    if not pd_.empty:
        mine = pd_[ (pd_["a1"].eq(player_id)) | (pd_["a2"].eq(player_id)) | (pd_["b1"].eq(player_id)) | (pd_["b2"].eq(player_id)) ]
        if not mine.empty:
            st.markdown("**Doppel**")
            for _, r in mine.iterrows():
                a1 = id_to_name.get(r["a1"], r["a1"]); a2 = id_to_name.get(r["a2"], r["a2"])
                b1 = id_to_name.get(r["b1"], r["b1"]); b2 = id_to_name.get(r["b2"], r["b2"])
                cols = st.columns([3,1,1])
                cols[0].write(f"{a1}/{a2} vs {b1}/{b2}  {int(r['punktea'])}:{int(r['punkteb'])}")
                if cols[1].button("✅", key=f"conf_d_{r['id']}"):
                    confirm_double(r)
                    supabase.table("pending_doubles").delete().eq("id", r["id"]).execute()
                    clear_table_cache()
                    st.success("Doppel bestätigt.")
                    st.experimental_rerun()
                if cols[2].button("❌", key=f"rej_d_{r['id']}"):
                    supabase.table("pending_doubles").delete().eq("id", r["id"]).execute()
                    clear_table_cache()
                    st.info("Doppel verworfen.")
                    st.experimental_rerun()

    # Rundlauf
    pr = load_table("pending_rounds")
    if not pr.empty:
        # Zeige alle, an denen der Spieler beteiligt ist (steht als ID in teilnehmer oder finalisten oder sieger)
        def involved(r: pd.Series) -> bool:
            joined = ";".join([str(r.get("teilnehmer","")), str(r.get("finalisten","")), str(r.get("sieger",""))])
            return str(player_id) in joined
        mine = pr[pr.apply(involved, axis=1)]
        if not mine.empty:
            st.markdown("**Rundlauf**")
            for _, r in mine.iterrows():
                teiln = [id_to_name.get(pid, pid) for pid in str(r["teilnehmer"]).split(";") if pid]
                sieger = id_to_name.get(r.get("sieger"), r.get("sieger"))
                cols = st.columns([3,1,1])
                cols[0].write(f"{', '.join(teiln)}  Sieger: {sieger}")
                if cols[1].button("✅", key=f"conf_r_{r['id']}"):
                    confirm_round(r)
                    supabase.table("pending_rounds").delete().eq("id", r["id"]).execute()
                    clear_table_cache()
                    st.success("Rundlauf bestätigt.")
                    st.experimental_rerun()
                if cols[2].button("❌", key=f"rej_r_{r['id']}"):
                    supabase.table("pending_rounds").delete().eq("id", r["id"]).execute()
                    clear_table_cache()
                    st.info("Rundlauf verworfen.")
                    st.experimental_rerun()

# ---- Rangliste ----
def leaderboard_ui():
    st.subheader("Rangliste")
    players = load_table("players")
    if players.empty:
        st.info("Noch keine Spieler.")
        return
    # Anzeige nach Gesamt-ELO
    cols_show = ["name", "g_elo", "elo", "d_elo", "r_elo", "spiele", "siege", "niederlagen"]
    for c in cols_show:
        if c not in players.columns:
            players[c] = np.nan
    df = players[cols_show].copy()
    df = df.sort_values("g_elo", ascending=False, na_position="last")
    st.dataframe(df, use_container_width=True)

# ---- Startseite / Letzte Matches ----
def home_ui(user: dict):
    header(user)
    # Letzte 5 Matches (modus-unabhängig)
    matches = load_table("matches")
    doubles = load_table("doubles")
    rounds = load_table("rounds")
    id_to_name, _ = get_player_maps()

    combined = []
    if not matches.empty:
        m = matches[(matches["a"].eq(user["id"])) | (matches["b"].eq(user["id"]))].copy()
        if not m.empty:
            m["modus"] = "Einzel"
            def opp(row):
                oid = row["b"] if row["a"] == user["id"] else row["a"]
                return id_to_name.get(oid, oid)
            m["gegner"] = m.apply(opp, axis=1)
            m["ergebnis"] = m.apply(lambda r: f"{int(r['punktea'])}:{int(r['punkteb'])}", axis=1)
            m["win"] = m.apply(lambda r: (r["punktea"] > r["punkteb"]) if r["a"]==user["id"] else (r["punkteb"] > r["punktea"]), axis=1)
            combined.append(m[["datum","modus","gegner","ergebnis","win"]])
    if not doubles.empty:
        d = doubles[(doubles[["a1","a2","b1","b2"]]==user["id"]).any(axis=1)].copy()
        if not d.empty:
            d["modus"] = "Doppel"
            def d_opp(row):
                opps = [row[x] for x in ["a1","a2","b1","b2"] if row[x] != user["id"]]
                # Gegner sind immer die andere Paarung
                if user["id"] in (row["a1"], row["a2"]):
                    opp_ids = [row["b1"], row["b2"]]
                else:
                    opp_ids = [row["a1"], row["a2"]]
                return "/".join([id_to_name.get(x,x) for x in opp_ids])
            d["gegner"] = d.apply(d_opp, axis=1)
            d["ergebnis"] = d.apply(lambda r: f"{int(r['punktea'])}:{int(r['punkteb'])}", axis=1)
            def d_win(r):
                if user["id"] in (r["a1"], r["a2"]):
                    return r["punktea"] > r["punkteb"]
                return r["punkteb"] > r["punktea"]
            d["win"] = d.apply(d_win, axis=1)
            combined.append(d[["datum","modus","gegner","ergebnis","win"]])
    if not rounds.empty:
        r = rounds.copy()
        def r_involved(x):
            return str(user["id"]) in str(x)
        r = r[r["teilnehmer"].apply(r_involved)]
        if not r.empty:
            r["modus"] = "Rundlauf"
            r["gegner"] = r["teilnehmer"].apply(lambda s: ", ".join([id_to_name.get(pid, pid) for pid in str(s).split(";") if pid]))
            r["ergebnis"] = r["sieger"].apply(lambda x: f"Sieger: {id_to_name.get(x, x)}")
            r["win"] = r["sieger"] == user["id"]
            combined.append(r[["datum","modus","gegner","ergebnis","win"]])

    if combined:
        last5 = pd.concat(combined, ignore_index=True).sort_values("datum", ascending=False).head(5)
        st.markdown("### Letzte Spiele")
        st.table(last5)

# ---- Main ----
def main():
    st.markdown("""
        <style>
            .stMetric { text-align:center; }
        </style>
    """, unsafe_allow_html=True)

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.player_id = None
        st.session_state.player_name = None

    if not st.session_state.logged_in:
        login_ui()
        return

    # Hole User-Record frisch
    user = supabase.table("players").select("*").eq("id", st.session_state.player_id).single().execute().data
    if not user:
        st.error("Nutzer nicht gefunden – bitte erneut einloggen.")
        for k in ("logged_in","player_id","player_name"):
            st.session_state.pop(k, None)
        st.rerun()
        return

    tabs = st.tabs(["Home", "Eintragen", "Bestätigen", "Rangliste", "Profil"])    
    with tabs[0]:
        home_ui(user)
    with tabs[1]:
        create_invite_ui(user["id"])
    with tabs[2]:
        confirm_ui(user["id"])
    with tabs[3]:
        leaderboard_ui()
    with tabs[4]:
        st.write(f"Eingeloggt als **{user['name']}**")
        if st.button("Logout"):
            for k in ("logged_in","player_id","player_name"):
                st.session_state.pop(k, None)
            st.rerun()

if __name__ == "__main__":
    main()
