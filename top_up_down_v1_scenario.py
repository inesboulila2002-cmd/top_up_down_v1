import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="miRNA Predictor — v1 Scenario", page_icon="🧬", layout="wide")
st.title("🧬 miRNA Upregulation Predictor — v1 Scenario")
st.caption("Random Forest · OHE · Features: miRNA group, parasite, organism, cell type, time, scenario · OOB evaluation")

ORGANISM_PREFIX = {"Human": "hsa-", "Mouse": "mmu-"}

# ── Load model ────────────────────────────────────────────────────────────────
# v1 was saved as a raw sklearn pipeline (not a dict bundle).
# We borrow mirna_lookup from v2 or v3 since they share the same dataset.
V1_MODEL_FILE = "Mir_v1_scenario_model.pkl"
FALLBACK_FILES = ["Mir_v2_family_scenario_model.pkl", "Mir_v3_family_model.pkl"]

@st.cache_resource
def load_v1():
    if not os.path.exists(V1_MODEL_FILE):
        return None, None, None

    pipeline = joblib.load(V1_MODEL_FILE)

    # If someone already saved v1 as a dict bundle, handle that too
    if isinstance(pipeline, dict):
        return (
            pipeline['model'],
            pipeline.get('mirna_lookup', {}),
            pipeline.get('oob_score', None),
        )

    # Raw pipeline — borrow mirna_lookup from v2 or v3
    mirna_lookup = {}
    for fallback in FALLBACK_FILES:
        if os.path.exists(fallback):
            bundle = joblib.load(fallback)
            if isinstance(bundle, dict) and 'mirna_lookup' in bundle:
                mirna_lookup = bundle['mirna_lookup']
                break

    # OOB score is stored inside the classifier step
    try:
        oob_score = pipeline.named_steps['classifier'].oob_score_
    except Exception:
        oob_score = None

    return pipeline, mirna_lookup, oob_score

model, mirna_lookup, oob_score = load_v1()

if model is None:
    st.error(f"Model file `{V1_MODEL_FILE}` not found. Place it in the same directory as this script.")
    st.stop()

if not mirna_lookup:
    st.error(
        f"No miRNA lookup found. Place at least one of "
        f"`{'` or `'.join(FALLBACK_FILES)}` alongside this script so miRNA names can be loaded."
    )
    st.stop()

# ── OOB banner ────────────────────────────────────────────────────────────────
if oob_score:
    st.markdown("#### Model Performance")
    st.metric("OOB Accuracy", f"{oob_score:.3f}")
    st.divider()

# ── Helper ────────────────────────────────────────────────────────────────────
def build_input_row(mirna_group, parasite, organism, cell_type, time_val):
    return pd.DataFrame([{
        'microrna_group_simplified': mirna_group,
        'parasite':                  parasite,
        'organism':                  organism,
        'cell type':                 cell_type,
        'time':                      int(time_val),
        'scenario':                  f"{parasite.strip()}_{cell_type.strip()}",
    }])

# ── UI ────────────────────────────────────────────────────────────────────────
st.subheader("📊 Top miRNAs Predicted Up or Down Under Your Conditions")
st.caption("Scores every miRNA in the database that matches the selected organism.")

col_a, col_b = st.columns(2)
rl_parasite = col_a.selectbox("Parasite",  ["L.major", "L.donovani", "L.amazonensis", "L. donovani"])
rl_organism = col_b.selectbox("Organism",  ["Human", "Mouse"])
rl_cell     = col_a.selectbox("Cell type", ["PBMC", "THP-1", "BMDM (BALB/c females)", "RAW 264.7",
                                             "Blood serum + liver (BALB/c )"])
rl_time     = col_b.number_input("Time (hours post-infection)", min_value=1, value=24)
rl_top_n    = st.slider("Number of miRNAs to show per direction", 5, 30, 10)

if st.button("🔍 Rank All miRNAs", type="primary"):
    prefix = ORGANISM_PREFIX.get(rl_organism, "")

    # ── Organism filter ───────────────────────────────────────────────────────
    filtered = {
        name: entry
        for name, entry in mirna_lookup.items()
        if not prefix or name.lower().startswith(prefix.lower())
    }

    if not filtered:
        st.warning(f"No `{prefix}` miRNAs found in the database for organism **{rl_organism}**.")
        st.stop()

    rows = []
    for mirna_name, entry in filtered.items():
        mirna_group = entry.get('microrna_group_simplified', mirna_name)
        input_df    = build_input_row(mirna_group, rl_parasite, rl_organism, rl_cell, rl_time)
        proba       = model.predict_proba(input_df)[0][1]

        rows.append({
            "miRNA":          mirna_name,
            "Parasite":       rl_parasite,
            "Cell Type":      rl_cell,
            "Time (h)":       int(rl_time),
            "Direction":      "⬆️ Up" if proba >= 0.5 else "⬇️ Down",
            "P(up)":          proba,
            "Confidence (%)": round(max(proba, 1 - proba) * 100, 1),
        })

    df_all = pd.DataFrame(rows)

    top_up = (df_all[df_all["P(up)"] >= 0.5]
              .sort_values("P(up)", ascending=False)
              .head(rl_top_n).reset_index(drop=True))
    top_up.index += 1

    top_down = (df_all[df_all["P(up)"] < 0.5]
                .sort_values("P(up)", ascending=True)
                .head(rl_top_n).reset_index(drop=True))
    top_down.index += 1

    DISPLAY_COLS = ["miRNA", "Parasite", "Cell Type", "Time (h)", "Confidence (%)"]

    st.markdown(f"**Conditions:** {rl_parasite} · {rl_organism} · {rl_cell} · {rl_time}h")
    st.markdown(f"Scored **{len(df_all)} `{prefix}` miRNAs** — showing top {rl_top_n} per direction.")
    st.divider()

    col_up, col_down = st.columns(2)
    with col_up:
        st.markdown("### ⬆️ Top Upregulated")
        if top_up.empty:
            st.info("No miRNAs predicted as upregulated.")
        else:
            st.dataframe(top_up[DISPLAY_COLS], use_container_width=True)
    with col_down:
        st.markdown("### ⬇️ Top Downregulated")
        if top_down.empty:
            st.info("No miRNAs predicted as downregulated.")
        else:
            st.dataframe(top_down[DISPLAY_COLS], use_container_width=True)
