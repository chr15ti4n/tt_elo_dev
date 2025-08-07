 import streamlit as st
from supabase import create_client, Client
from streamlit_ketcher import st_ketcher

# ---------- Supabase Verbindung einrichten ----------
@st.cache_resource
def init_connection() -> Client:
    """Stellt die Verbindung zur Supabase-Datenbank her (wird gecached, um nicht bei jedem Rerun neu zu verbinden)."""
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()

# Hilfsfunktionen für Datenbank-Operationen
def get_all_data():
    try:
        resp_sub = supabase.table("substances").select("*").execute()
        resp_att = supabase.table("attempts").select("*").execute()
        resp_frac= supabase.table("fractions").select("*").execute()
    except Exception as e:
        st.error(f"Datenbank-Fehler: {e}")
        return [], {}, {}

    # Statt .error zu prüfen, greifst du auf .data zu:
    substances = resp_sub.data or []
    attempts   = resp_att.data   or []
    fractions  = resp_frac.data  or []

    # Jetzt wie gehabt verschachteln:
    attempts_by_substance = {}
    for att in attempts:
        attempts_by_substance.setdefault(att["substance_id"], []).append(att)
    fractions_by_attempt = {}
    for frac in fractions:
        fractions_by_attempt.setdefault(frac["attempt_id"], []).append(frac)
    return substances, attempts_by_substance, fractions_by_attempt

def add_substance(smiles: str, name: str = ""):
    """Fügt eine neue Substanz mit gegebener Struktur (SMILES) und optionalem Namen hinzu. Vergibt automatisch einen CHKN-Code."""
    # Ermittle nächste CHKN Nummer:
    # Wir nehmen an, der code soll "CHKN" gefolgt von der nächsten freien Nummer sein.
    # Dies ist eine einfache Variante: nimm die Anzahl vorhandener Substanzen + 1 als neue Nummer.
    # (Alternativ könnte man auch MAX(id) abfragen. Hier vereinfachend:)
    resp_count = supabase.table("substances").select("id", count="exact").execute()
    new_number = (resp_count.count or 0) + 1
    new_code = f"CHKN{new_number}"
    data = {"code": new_code, "structure_smiles": smiles}
    if name:
        data["name"] = name
    # Eintrag in DB erstellen
    resp = supabase.table("substances").insert(data).execute()
    if resp.error:
        st.error(f"Fehler beim Hinzufügen der Substanz: {resp.error}")
    return resp

def delete_substance(substance_id: int):
    """Löscht eine Substanz (und cascaded automatisch zugehörige Ansätze/Fraktionen)."""
    resp = supabase.table("substances").delete().eq("id", substance_id).execute()
    if resp.error:
        st.error(f"Konnte Substanz nicht löschen: {resp.error}")

def add_attempt(substance_id: int, notes: str, status: str, theoretical_yield: float = None):
    """Fügt einen neuen Reaktionsansatz zu einer Substanz hinzu."""
    # Nächste Ansatz-Nummer ermitteln für die Substanz:
    # Hier: zähle existierende Ansätze der Substanz und +1
    resp_count = supabase.table("attempts").select("id", count="exact").eq("substance_id", substance_id).execute()
    next_no = (resp_count.count or 0) + 1
    data = {"substance_id": substance_id, "attempt_no": next_no, "notes": notes, "status": status}
    if theoretical_yield is not None and theoretical_yield != "":
        # theoretisches Feld als Zahl speichern
        try:
            data["theoretical_yield_mg"] = float(theoretical_yield)
        except:
            st.warning("Theoretische Ausbeute konnte nicht als Zahl interpretiert werden.")
    resp = supabase.table("attempts").insert(data).execute()
    if resp.error:
        st.error(f"Fehler beim Hinzufügen des Ansatzes: {resp.error}")

def delete_attempt(attempt_id: int):
    """Löscht einen Reaktionsansatz (und cascaded zugehörige Fraktionen)."""
    resp = supabase.table("attempts").delete().eq("id", attempt_id).execute()
    if resp.error:
        st.error(f"Konnte Ansatz nicht löschen: {resp.error}")

def add_fraction(attempt_id: int, yield_mg: float, purity: float, analyses: dict, is_final: bool):
    """Fügt eine neue Fraktion hinzu. 'analyses' ist ein Dict mit booleans für die Analysen-Checkboxen."""
    # Nächste Fraktionsnummer ermitteln:
    resp_count = supabase.table("fractions").select("id", count="exact").eq("attempt_id", attempt_id).execute()
    next_no = (resp_count.count or 0) + 1
    # Daten vorbereiten
    data = {"attempt_id": attempt_id, "fraction_no": next_no}
    # Optional-Felder nur setzen, wenn Werte angegeben
    if yield_mg not in (None, ""):
        try:
            data["yield_mg"] = float(yield_mg)
        except:
            st.warning("Ausbeute (mg) wird nicht als Zahl erkannt und wurde ignoriert.")
    if purity not in (None, ""):
        try:
            data["purity_percent"] = float(purity)
        except:
            st.warning("Reinheit (%) wird nicht als Zahl erkannt und wurde ignoriert.")
    # Analysen-Checkboxen eintragen:
    data.update({
        "analysis_eims": analyses.get("eims", False),
        "analysis_1h_nmr": analyses.get("1h_nmr", False),
        "analysis_13c_nmr": analyses.get("13c_nmr", False),
        "analysis_elem": analyses.get("elem", False),
        "analysis_mp": analyses.get("mp", False)
    })
    # Final-Logik: Nur erlauben, wenn EI-MS und 1H-NMR vorhanden
    if is_final:
        if not (data["analysis_eims"] and data["analysis_1h_nmr"]):
            st.error("Diese Fraktion kann nicht als 'final' markiert werden, solange EI-MS und ¹H-NMR nicht beide durchgeführt wurden.")
            is_final = False  # override to False, da Validierung nicht erfüllt
        else:
            data["is_final"] = True
    resp = supabase.table("fractions").insert(data).execute()
    if resp.error:
        st.error(f"Fehler beim Hinzufügen der Fraktion: {resp.error}")
    else:
        # Wenn erfolgreich hinzugefügt und als final markiert, Ansatz-Status auf 'erfolgreich' setzen
        if is_final:
            supabase.table("attempts").update({"status": "erfolgreich"}).eq("id", attempt_id).execute()

def delete_fraction(frac_id: int):
    """Löscht eine Fraktion."""
    resp = supabase.table("fractions").delete().eq("id", frac_id).execute()
    if resp.error:
        st.error(f"Konnte Fraktion nicht löschen: {resp.error}")

# ---------- UI Rendering ----------
st.title("📒 Chemisches Reaktionstracking")
st.write("Diese Anwendung erlaubt das Verfolgen von Substanzen, Reaktionsansätzen und Fraktionen während deiner Laborarbeit.")

# 1. Formular zum Hinzufügen einer neuen Substanz
st.header("Neue Substanz hinzufügen")
with st.form("new_substance_form", clear_on_submit=True):
    st.markdown("**Struktur zeichnen / eingeben:**")
    # Molekül-Editor (Ketcher) Komponente zum Zeichnen; gibt SMILES zurück [oai_citation:0‡blog.streamlit.io](https://blog.streamlit.io/introducing-a-chemical-molecule-component-for-your-streamlit-apps/#:~:text=pip%20install%20streamlit%20pip%20install,ketcher)
    smiles = st_ketcher()
    st.caption("Der obige Moleküleditor ermöglicht das Zeichnen einer Struktur im Browser und gibt einen SMILES-String zurück [oai_citation:1‡blog.streamlit.io](https://blog.streamlit.io/introducing-a-chemical-molecule-component-for-your-streamlit-apps/#:~:text=import%20streamlit%20as%20st%20from,streamlit_ketcher%20import%20st_ketcher).")
    name_input = st.text_input("Name der Substanz (optional, z.B. IUPAC-Name)", "")
    submitted = st.form_submit_button("➕ Substanz anlegen")
    if submitted:
        if not smiles or smiles.strip() == "":
            st.error("Bitte zeichne eine chemische Struktur, bevor du die Substanz anlegst.")
        else:
            res = add_substance(smiles, name_input.strip())
            if res and not res.error:
                st.success("Substanz erfolgreich hinzugefügt!")
                st.experimental_rerun()  # UI aktualisieren

# 2. Übersicht der vorhandenen Substanzen
st.header("Substanzen")
substances, attempts_by_substance, fractions_by_attempt = get_all_data()
if len(substances) == 0:
    st.info("Noch keine Substanzen vorhanden. Lege oben eine neue Substanz an.")
for sub in substances:
    sub_id = sub["id"]
    sub_code = sub.get("code", f"Substanz {sub_id}")
    sub_name = sub.get("name")
    # Anzeige der Substanz (Code und Name)
    sub_header = f"**{sub_code}**"
    if sub_name:
        sub_header += f" – {sub_name}"
    # Optional: Strukturbild anzeigen (hier könnten wir z.B. mittels RDKit aus SMILES ein Bild erzeugen)
    # For simplicity, we just show the SMILES as text or a placeholder.
    sub_header += f"  \nSMILES: `{sub.get('structure_smiles', '')}`"
    with st.expander(sub_header, expanded=False):
        # Lösch-Button für Substanz
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            st.write("**Ansätze für diese Substanz:**")
        with col2:
            if st.button("🗑️ Substanz löschen", key=f"del_sub_{sub_id}"):
                delete_substance(sub_id)
                st.warning(f"Substanz {sub_code} gelöscht.")
                st.experimental_rerun()
        # Ansätze auflisten
        sub_attempts = attempts_by_substance.get(sub_id, [])
        for att in sorted(sub_attempts, key=lambda x: x["attempt_no"]):
            att_id = att["id"]
            att_code = f"{sub_code}-{att['attempt_no']}"
            status = att.get("status", "laufend")
            status_color = {"laufend": "🟡 laufend", "erfolgreich": "🟢 erfolgreich", "fehlgeschlagen": "🔴 fehlgeschlagen"}.get(status, status)
            notes = att.get("notes", "")
            theoretical = att.get("theoretical_yield_mg")
            header = f"**Ansatz {att_code}** – Status: *{status_color}*"
            if notes:
                header += "  \n📝 " + notes.replace("\n", " ")
            if theoretical:
                header += f"  \n(Theoretische Ausbeute: {theoretical} mg)"
            with st.expander(header, expanded=False):
                # Lösch-Button für Ansatz
                colA, colB = st.columns([0.9, 0.1])
                with colA:
                    st.write("**Fraktionen dieses Ansatzes:**")
                with colB:
                    if st.button("🗑️ Ansatz löschen", key=f"del_att_{att_id}"):
                        delete_attempt(att_id)
                        st.warning(f"Ansatz {att_code} gelöscht.")
                        st.experimental_rerun()
                # Fraktionen auflisten
                att_fractions = fractions_by_attempt.get(att_id, [])
                for frac in sorted(att_fractions, key=lambda x: x["fraction_no"]):
                    frac_id = frac["id"]
                    frac_code = f"{att_code}-{frac['fraction_no']}"
                    yield_mg = frac.get("yield_mg")
                    purity = frac.get("purity_percent")
                    is_final = frac.get("is_final", False)
                    # Berechne Ausbeute in % falls möglich
                    yield_pct = None
                    if yield_mg is not None and purity is not None and theoretical:
                        # pure Ausbeute = yield_mg * (purity/100)
                        try:
                            yield_pct = (float(yield_mg) * float(purity) / 100.0) / float(theoretical) * 100.0
                        except:
                            yield_pct = None
                    # Analysen-Status
                    analyses_done = []
                    if frac.get("analysis_eims"): analyses_done.append("EI-MS")
                    if frac.get("analysis_1h_nmr"): analyses_done.append("¹H-NMR")
                    if frac.get("analysis_13c_nmr"): analyses_done.append("¹³C-NMR")
                    if frac.get("analysis_elem"): analyses_done.append("Elementaranalyse")
                    if frac.get("analysis_mp"): analyses_done.append("MP")
                    analyses_text = ", ".join(analyses_done) if analyses_done else "Keine Analytik abgeschlossen"
                    final_mark = "✅ FINAL" if is_final else ""
                    # Fraktion Zeile anzeigen
                    frac_info = f"**Fraktion {frac_code}:** "
                    if yield_mg is not None:
                        frac_info += f"Ausbeute {yield_mg} mg"
                        if purity is not None:
                            frac_info += f", Reinheit {purity}%"
                        if yield_pct is not None:
                            frac_info += f" (→ {yield_pct:.1f}% Ausbeute)"
                        frac_info += "; "
                    frac_info += f"Analytik: {analyses_text}"
                    if final_mark:
                        frac_info += f" – {final_mark}"
                    st.write(frac_info)
                    # Lösch-Button für Fraktion
                    if st.button("löschen", key=f"del_frac_{frac_id}"):
                        delete_fraction(frac_id)
                        st.info(f"Fraktion {frac_code} gelöscht.")
                        st.experimental_rerun()
                # Formular zum Hinzufügen einer neuen Fraktion unter diesem Ansatz
                st.subheader(f"Neue Fraktion zu {att_code} hinzufügen")
                with st.form(f"new_fraction_form_{att_id}", clear_on_submit=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        yield_input = st.text_input("Ausbeute (mg)")
                        purity_input = st.text_input("Reinheit (%)")
                        final_input = st.checkbox("Als final markieren?")
                    with col2:
                        st.markdown("Durchgeführte Analysen:")
                        eims = st.checkbox("EI-MS", value=False)
                        hnmr = st.checkbox("¹H-NMR", value=False)
                        cnmr = st.checkbox("¹³C-NMR", value=False)
                        elem = st.checkbox("Elementaranalyse", value=False)
                        mp   = st.checkbox("Schmelzpunkt (MP)", value=False)
                    add_frac_btn = st.form_submit_button("➕ Fraktion anlegen")
                    if add_frac_btn:
                        analyses = {"eims": eims, "1h_nmr": hnmr, "13c_nmr": cnmr, "elem": elem, "mp": mp}
                        add_fraction(att_id, yield_input, purity_input, analyses, final_input)
                        # Bei Erfolg wird in add_fraction bereits ggf. status aktualisiert
                        st.success(f"Fraktion hinzugefügt{' (final)' if final_input and eims and hnmr else ''}!")
                        st.experimental_rerun()
        # Formular zum Hinzufügen eines neuen Ansatzes unter der Substanz
        st.subheader(f"Neuen Reaktionsansatz für {sub_code} hinzufügen")
        with st.form(f"new_attempt_form_{sub_id}", clear_on_submit=True):
            notes_input = st.text_area("Reaktionsbedingungen / Notizen")
            status_input = st.selectbox("Status", options=["laufend", "erfolgreich", "fehlgeschlagen"], index=0)
            theo_input = st.text_input("Theoretische Ausbeute (mg, optional)")
            submit_attempt = st.form_submit_button("➕ Ansatz anlegen")
            if submit_attempt:
                add_attempt(sub_id, notes_input, status_input, theo_input)
                st.success(f"Ansatz hinzugefügt (Status: {status_input}).")
                st.experimental_rerun()
