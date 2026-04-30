import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import time

st.set_page_config(
    page_title="ChemClassify AI",
    page_icon="⚗️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600&display=swap');

:root {
    --neon-cyan:   #00f5ff;
    --neon-pink:   #ff006e;
    --neon-green:  #39ff14;
    --neon-purple: #bf5af2;
    --dark-bg:     #050510;
    --card-bg:     #0a0a1a;
    --border:      rgba(0,245,255,0.25);
}
html, body, [data-testid="stAppViewContainer"] { background: var(--dark-bg) !important; }
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 40% at 20% 0%,   rgba(0,245,255,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 100%, rgba(191,90,242,0.07) 0%, transparent 60%),
        var(--dark-bg) !important;
    background-attachment: fixed !important;
}
[data-testid="stHeader"]  { background: transparent !important; }
section.main > div { padding-top: 1rem !important; }
h1,h2,h3 { font-family:'Orbitron',monospace !important; }
p,label,div { font-family:'Rajdhani',sans-serif !important; }

.hero-header { text-align:center; padding:2.5rem 1rem 1.5rem; }
.hero-title {
    font-family:'Orbitron',monospace !important;
    font-size:clamp(2rem,6vw,3.5rem); font-weight:900; letter-spacing:0.08em;
    background:linear-gradient(135deg,var(--neon-cyan) 0%,var(--neon-purple) 50%,var(--neon-pink) 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    margin:0; line-height:1.1;
}
.hero-sub {
    font-family:'Share Tech Mono',monospace; color:rgba(0,245,255,0.65);
    font-size:0.9rem; letter-spacing:0.2em; margin-top:0.5rem; text-transform:uppercase;
}
.hero-line {
    width:60%; height:1px; margin:1.2rem auto 0;
    background:linear-gradient(90deg,transparent,var(--neon-cyan),var(--neon-pink),transparent);
}
.neon-card {
    background:var(--card-bg); border:1px solid var(--border); border-radius:16px;
    padding:1.8rem; margin:1rem 0;
    box-shadow:0 0 20px rgba(0,245,255,0.05),0 0 60px rgba(0,245,255,0.03),inset 0 1px 0 rgba(0,245,255,0.1);
    position:relative; overflow:hidden;
}
.neon-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent,var(--neon-cyan),transparent);
}
.section-label {
    font-family:'Share Tech Mono',monospace; font-size:0.72rem;
    letter-spacing:0.25em; color:var(--neon-cyan); text-transform:uppercase;
    margin-bottom:0.6rem; opacity:0.8;
}
textarea {
    background:#080818 !important; border:1px solid rgba(0,245,255,0.3) !important;
    border-radius:10px !important; color:#e0f0ff !important;
    font-family:'Rajdhani',sans-serif !important; font-size:1.05rem !important;
    caret-color:var(--neon-cyan) !important; resize:vertical !important;
}
textarea:focus {
    border-color:var(--neon-cyan) !important;
    box-shadow:0 0 0 2px rgba(0,245,255,0.15),0 0 25px rgba(0,245,255,0.1) !important;
    outline:none !important;
}
textarea::placeholder { color:rgba(0,245,255,0.3) !important; }
[data-testid="stTextArea"] label { display:none !important; }
[data-testid="stTextArea"] > div:first-child { height:0 !important; overflow:hidden !important; }

.stButton > button {
    width:100%;
    background:linear-gradient(135deg,rgba(0,245,255,0.12),rgba(191,90,242,0.12)) !important;
    border:1px solid var(--neon-cyan) !important; border-radius:10px !important;
    color:var(--neon-cyan) !important; font-family:'Orbitron',monospace !important;
    font-size:0.88rem !important; font-weight:700 !important;
    letter-spacing:0.18em !important; padding:0.75rem 1.5rem !important;
    text-transform:uppercase !important;
    box-shadow:0 0 15px rgba(0,245,255,0.12),inset 0 1px 0 rgba(0,245,255,0.15) !important;
    transition:all 0.25s ease !important;
}
.stButton > button:hover {
    background:linear-gradient(135deg,rgba(0,245,255,0.25),rgba(191,90,242,0.25)) !important;
    box-shadow:0 0 30px rgba(0,245,255,0.3),0 0 60px rgba(0,245,255,0.1) !important;
    transform:translateY(-1px) !important;
}

/* ── Chip styles — pure HTML, fully under our control ── */
.chip-wrap {
    display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.5rem;
}
.chip {
    display:inline-block;
    background:rgba(0,245,255,0.06);
    border:1px solid rgba(0,245,255,0.28);
    border-radius:100px;
    color:rgba(0,245,255,0.78) !important;
    font-family:'Rajdhani',sans-serif;
    font-size:0.83rem; font-weight:600;
    letter-spacing:0.03em;
    padding:0.35rem 1rem;
    text-decoration:none !important;
    cursor:pointer;
    transition:all 0.2s ease;
    white-space:nowrap;
}
.chip:hover {
    background:rgba(0,245,255,0.16);
    border-color:#00f5ff;
    color:#00f5ff !important;
    box-shadow:0 0 14px rgba(0,245,255,0.25);
    transform:translateY(-1px);
    text-decoration:none !important;
}

.stats-row { display:flex; gap:1rem; margin:1rem 0; }
.stat-cell { flex:1; background:rgba(0,245,255,0.05); border:1px solid rgba(0,245,255,0.15); border-radius:10px; padding:0.85rem; text-align:center; }
.stat-val  { font-family:'Orbitron',monospace; font-size:1.4rem; font-weight:700; color:var(--neon-cyan); }
.stat-lbl  { font-family:'Share Tech Mono',monospace; font-size:0.65rem; letter-spacing:0.15em; color:rgba(0,245,255,0.45); text-transform:uppercase; margin-top:0.2rem; }

.result-box {
    background:linear-gradient(135deg,rgba(0,245,255,0.06),rgba(191,90,242,0.06));
    border:1px solid rgba(0,245,255,0.35); border-radius:14px;
    padding:1.5rem 1.8rem; margin:1.2rem 0; position:relative; overflow:hidden;
    animation:fadeSlideIn 0.5s ease forwards;
}
.result-box::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,var(--neon-cyan),var(--neon-purple),var(--neon-pink)); }
@keyframes fadeSlideIn { from{opacity:0;transform:translateY(12px);}to{opacity:1;transform:translateY(0);} }
.result-label { font-family:'Share Tech Mono',monospace; font-size:0.7rem; letter-spacing:0.3em; color:rgba(0,245,255,0.55); text-transform:uppercase; margin-bottom:0.4rem; }
.result-topic { font-family:'Orbitron',monospace; font-size:clamp(1.2rem,4vw,1.8rem); font-weight:700; background:linear-gradient(135deg,var(--neon-cyan),var(--neon-purple)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; letter-spacing:0.04em; line-height:1.2; }
.result-conf  { font-family:'Share Tech Mono',monospace; color:var(--neon-green); font-size:0.82rem; margin-top:0.5rem; letter-spacing:0.1em; }
.conf-bar-wrap { margin-top:0.8rem; background:rgba(255,255,255,0.05); border-radius:100px; height:6px; overflow:hidden; }
.conf-bar-fill { height:100%; border-radius:100px; background:linear-gradient(90deg,var(--neon-cyan),var(--neon-purple)); box-shadow:0 0 10px rgba(0,245,255,0.5); }

.top5-row { display:flex; align-items:center; gap:0.75rem; padding:0.55rem 0.4rem; border-bottom:1px solid rgba(0,245,255,0.07); }
.top5-row:last-child { border-bottom:none; }
.top5-rank { font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:rgba(0,245,255,0.4); width:1.6rem; flex-shrink:0; }
.top5-name { font-family:'Rajdhani',sans-serif; font-size:0.95rem; font-weight:600; color:#c0d8f0; flex:1; }
.top5-pct  { font-family:'Share Tech Mono',monospace; font-size:0.78rem; color:var(--neon-green); width:3.5rem; text-align:right; flex-shrink:0; }
.top5-mini-bar  { width:80px; height:4px; background:rgba(255,255,255,0.06); border-radius:100px; overflow:hidden; flex-shrink:0; }
.top5-mini-fill { height:100%; border-radius:100px; background:linear-gradient(90deg,var(--neon-cyan),var(--neon-purple)); }

.err-box { background:rgba(255,0,110,0.08); border:1px solid rgba(255,0,110,0.35); border-radius:10px; padding:1rem 1.2rem; color:#ff7eb3; font-family:'Share Tech Mono',monospace; font-size:0.85rem; margin:0.8rem 0; }
.explain-box { background:linear-gradient(135deg,rgba(191,90,242,0.06),rgba(0,245,255,0.04)); border:1px solid rgba(191,90,242,0.3); border-radius:14px; padding:1.4rem 1.8rem; margin:0.8rem 0 1.2rem; position:relative; overflow:hidden; }
.explain-box::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,var(--neon-purple),var(--neon-pink),var(--neon-cyan)); }
.explain-label { font-family:'Share Tech Mono',monospace; font-size:0.7rem; letter-spacing:0.3em; color:rgba(191,90,242,0.7); text-transform:uppercase; margin-bottom:0.7rem; }
.explain-text  { font-family:'Rajdhani',sans-serif; font-size:1.05rem; color:#ccdaf0; line-height:1.7; }

.footer { text-align:center; padding:2rem 1rem 1rem; font-family:'Share Tech Mono',monospace; font-size:0.68rem; color:rgba(0,245,255,0.2); letter-spacing:0.15em; text-transform:uppercase; }
[data-testid="stSpinner"] > div { border-top-color:var(--neon-cyan) !important; }
::-webkit-scrollbar{width:6px;}::-webkit-scrollbar-track{background:var(--dark-bg);}::-webkit-scrollbar-thumb{background:rgba(0,245,255,0.25);border-radius:3px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = tf.keras.models.load_model("Chem_cnn_Model.keras")
    with open("tokenizer.pkl","rb") as f: tokenizer = pickle.load(f)
    with open("label_encoder.pkl","rb") as f: label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

def predict_topic(text, model, tokenizer, label_encoder, max_length=100):
    seq      = tokenizer.texts_to_sequences([text])
    padded   = pad_sequences(seq, maxlen=max_length, padding="post")
    probs    = model.predict(padded, verbose=0)[0]
    top5_idx = np.argsort(probs)[::-1][:5]
    top_class  = label_encoder.classes_[top5_idx[0]]
    confidence = float(probs[top5_idx[0]]) * 100
    top5 = [(label_encoder.classes_[i], float(probs[i])*100) for i in top5_idx]
    return top_class, confidence, top5

EXAMPLES = [
    "Explain the process of Lucas Test",
    "What is the rate constant in first-order reactions?",
    "How does hemoglobin carry oxygen in blood?",
    "Describe the mechanism of SN2 reaction",
    "What is Gibbs free energy?",
    "Explain coordination compounds and CFSE",
    "What is Heisenberg's uncertainty principle in chemistry?",
    "How do catalysts affect activation energy?",
]

TOPIC_EXPLANATIONS = {
    "Analytical chemistry":    "Analytical Chemistry is the science of identifying and measuring substances in a sample. It uses techniques like titration, chromatography, and spectroscopy to determine what a substance is and how much of it is present — used in quality control, medicine, and environmental testing.",
    "Biochemistry":            "Biochemistry explores the chemical processes happening inside living organisms. It covers how proteins fold, how DNA carries genetic information, how enzymes speed up reactions, and how energy is produced in cells — bridging chemistry and biology.",
    "Chemical bonding":        "Chemical Bonding explains how atoms connect to form molecules. It covers ionic bonds (electron transfer), covalent bonds (electron sharing), and intermolecular forces like hydrogen bonding — determining the shape, strength, and properties of every substance.",
    "Chemical education":      "Chemical Education focuses on how chemistry is taught and learned. It involves developing curricula, effective teaching strategies, laboratory safety, and making complex chemical concepts accessible to students at all levels.",
    "Chemical engineering":    "Chemical Engineering applies chemistry and physics to design industrial processes. It covers how to scale up reactions, design reactors, refine petroleum, manufacture pharmaceuticals, and process food — turning lab chemistry into real-world production.",
    "Chemical equilibrium":    "Chemical Equilibrium studies reactions that reach a balance where forward and reverse reactions occur at the same rate. It explains how changing temperature, pressure, or concentration shifts that balance — fundamental to industrial synthesis like making ammonia.",
    "Chemical kinetics":       "Chemical Kinetics is the study of reaction rates — how fast a chemical reaction happens and what factors control it. It covers rate laws, activation energy, reaction mechanisms, and how catalysts speed things up without being consumed.",
    "Chemical reactions":      "Chemical Reactions is the broad study of how substances transform into new substances by breaking and forming chemical bonds. It covers types of reactions like combustion, oxidation, neutralisation, and precipitation.",
    "Chemical synthesis":      "Chemical Synthesis is the process of building target molecules from simpler starting materials. It involves planning multi-step reaction sequences and is central to creating medicines, dyes, polymers, and new materials in the laboratory.",
    "Chemical thermodynamics": "Chemical Thermodynamics studies energy changes during chemical reactions. Using concepts like enthalpy, entropy, and Gibbs free energy, it predicts whether a reaction will occur spontaneously and how much energy it releases or absorbs.",
    "Computational chemistry": "Computational Chemistry uses mathematical models and computer simulations to study chemical systems. It predicts molecular structures, reaction pathways, and properties without needing physical experiments — essential in drug discovery and materials design.",
    "Coordination chemistry":  "Coordination Chemistry deals with metal ions bonded to surrounding molecules or ions called ligands. It explains the structure and colour of transition metal complexes and is important in catalysis, bioinorganic chemistry, and gemstone colour.",
    "Electrochemistry":        "Electrochemistry studies the relationship between electricity and chemical reactions. It underpins batteries, fuel cells, electroplating, corrosion, and electrolysis — explaining how chemical energy converts to electrical energy and vice versa.",
    "Environmental chemistry": "Environmental Chemistry examines how chemicals behave and interact in the environment — air, water, and soil. It covers pollution, toxicology, acid rain, ozone depletion, and the fate of pesticides and heavy metals in ecosystems.",
    "Inorganic chemistry":     "Inorganic Chemistry covers the properties and reactions of all elements and compounds that are not primarily carbon-based. It includes metals, minerals, salts, acids, bases, and transition metal chemistry — essentially everything outside organic molecules.",
    "Materials chemistry":     "Materials Chemistry focuses on designing and understanding the structure of materials to achieve specific properties. It covers semiconductors, nanomaterials, ceramics, composites, and surfaces — driving advances in electronics, energy, and construction.",
    "Medicinal chemistry":     "Medicinal Chemistry combines chemistry and pharmacology to design molecules that interact with biological targets. It explains how drugs are discovered, optimised for potency and safety, and how they interact with enzymes or receptors in the body.",
    "Nuclear chemistry":       "Nuclear Chemistry studies changes in atomic nuclei — radioactive decay, fission, fusion, and transmutation. It underpins nuclear power generation, radiation therapy for cancer, radiocarbon dating, and the science of radioisotopes.",
    "Organic chemistry":       "Organic Chemistry is the study of carbon-containing compounds and their reactions. It covers functional groups, reaction mechanisms like SN1/SN2, and the synthesis of everything from fuels and plastics to pharmaceuticals and natural products.",
    "Photochemistry":          "Photochemistry studies how light energy drives chemical reactions. When molecules absorb photons they reach excited states that trigger reactions — explaining photosynthesis, vitamin D production, photography, sunscreen action, and solar energy conversion.",
    "Physical chemistry":      "Physical Chemistry applies the principles of physics and mathematics to understand chemical systems. It covers quantum mechanics, thermodynamics, kinetics, spectroscopy, and statistical mechanics — providing the theoretical foundation for all chemistry.",
    "Polymer chemistry":       "Polymer Chemistry studies large chain-like molecules made of repeating units called monomers. It covers polymerisation reactions, polymer structure, and properties — explaining the science behind plastics, rubber, fibres, adhesives, and biological macromolecules like DNA.",
    "Quantum chemistry":       "Quantum Chemistry applies quantum mechanics to explain the behaviour of electrons in atoms and molecules. It describes atomic orbitals, molecular bonding, spectroscopy, and chemical reactivity at the most fundamental level using wave equations.",
    "Surface chemistry":       "Surface Chemistry examines chemical processes that occur at the interface between two phases — solid/liquid, solid/gas, or liquid/gas. It covers adsorption, catalysis, corrosion, and colloids — critical in industrial catalysts, detergents, and nanotechnology.",
    "Thermochemistry":         "Thermochemistry measures the heat energy exchanged during chemical reactions. Using calorimetry and Hess's law, it calculates reaction enthalpies — essential for understanding fuel energy content, industrial process design, and biological metabolism.",
}

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-title">⚗ ChemClassify AI</div>
  <div class="hero-sub">1D · CNN · Chemistry · Topic · Classifier</div>
  <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("Initialising neural network…"):
    try:
        model, tokenizer, label_encoder = load_artifacts()
        loaded_ok = True
    except Exception as e:
        loaded_ok = False; err_msg = str(e)

if not loaded_ok:
    st.markdown(f'<div class="err-box">⚠ Model load failed: {err_msg}</div>', unsafe_allow_html=True)
    st.stop()

num_classes = len(label_encoder.classes_)
st.markdown(f"""
<div class="stats-row">
  <div class="stat-cell"><div class="stat-val">{num_classes}</div><div class="stat-lbl">Topics</div></div>
  <div class="stat-cell"><div class="stat-val">1D-CNN</div><div class="stat-lbl">Architecture</div></div>
  <div class="stat-cell"><div class="stat-val">100</div><div class="stat-lbl">Max Tokens</div></div>
</div>
""", unsafe_allow_html=True)

# ── Session state + query param chip detection ────────────────────────────────
if "active_query" not in st.session_state:
    st.session_state.active_query = ""

# Read chip selection from query param, then clear it
params = st.query_params
if "chip" in params:
    idx = int(params["chip"])
    if 0 <= idx < len(EXAMPLES):
        st.session_state.active_query = EXAMPLES[idx]
    st.query_params.clear()
    st.rerun()

# ── Input card ────────────────────────────────────────────────────────────────
st.markdown('<div class="neon-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">◈ Enter your chemistry question</div>', unsafe_allow_html=True)

user_input = st.text_area(
    label="input",
    value=st.session_state.active_query,
    placeholder="e.g.  Explain the mechanism of nucleophilic addition to carbonyl groups…",
    height=130,
    key="chem_input",
)

predict_btn = st.button("⚡  CLASSIFY TOPIC", use_container_width=True)

# ── Example chips — pure HTML <a> tags, zero Streamlit involvement ─────────────
st.markdown('<div class="section-label" style="margin-top:1.4rem;">◈ Try an example</div>', unsafe_allow_html=True)

chip_html = '<div class="chip-wrap">'
for i, ex in enumerate(EXAMPLES):
    # Each chip is a plain anchor that sets ?chip=N, triggering a rerun above
    chip_html += f'<a class="chip" href="?chip={i}">{ex}</a>'
chip_html += '</div>'

st.markdown(chip_html, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # close neon-card

# ── Prediction ────────────────────────────────────────────────────────────────
text_to_classify = user_input.strip() if predict_btn else st.session_state.active_query.strip()
should_predict   = predict_btn or (st.session_state.active_query != "")

if should_predict:
    text = text_to_classify
    if not text:
        st.markdown('<div class="err-box">⚠ Please enter a chemistry question before classifying.</div>', unsafe_allow_html=True)
    elif len(text.split()) < 3:
        st.markdown('<div class="err-box">⚠ Query too short — please enter at least a few words.</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Analysing…"):
            time.sleep(0.3)
            topic, conf, top5 = predict_topic(text, model, tokenizer, label_encoder)

        conf_w = int(conf)
        st.markdown(f"""
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:rgba(0,245,255,0.4);letter-spacing:0.15em;margin:0.8rem 0 0.3rem;text-transform:uppercase;">◈ Classifying</div>
        <div style="font-family:'Rajdhani',sans-serif;font-size:1rem;color:#a0c8e8;background:rgba(0,245,255,0.04);border:1px solid rgba(0,245,255,0.12);border-radius:8px;padding:0.6rem 1rem;margin-bottom:0.5rem;">{text}</div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">◈ Predicted Chemistry Topic</div>
            <div class="result-topic">{topic}</div>
            <div class="result-conf">Confidence: {conf:.1f}%</div>
            <div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{conf_w}%"></div></div>
        </div>
        """, unsafe_allow_html=True)

        explanation = TOPIC_EXPLANATIONS.get(topic, "No explanation available.")
        st.markdown(f"""
        <div class="explain-box">
            <div class="explain-label">◈ What is {topic}?</div>
            <div class="explain-text">{explanation}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="neon-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">◈ Top-5 Prediction Breakdown</div>', unsafe_allow_html=True)
        max_p = top5[0][1] if top5 else 1
        for rank, (cls, pct) in enumerate(top5, 1):
            bar_w = int((pct / max_p) * 100)
            st.markdown(f"""
            <div class="top5-row">
                <div class="top5-rank">#{rank}</div>
                <div class="top5-name">{cls}</div>
                <div class="top5-mini-bar"><div class="top5-mini-fill" style="width:{bar_w}%"></div></div>
                <div class="top5-pct">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ── All topics ────────────────────────────────────────────────────────────────
with st.expander("📋  View all supported chemistry topics"):
    cols = st.columns(2)
    classes = list(label_encoder.classes_)
    half = len(classes)//2 + len(classes)%2
    for i, cls in enumerate(classes):
        col = cols[0] if i < half else cols[1]
        col.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif;font-size:0.9rem;color:#a0c8e8;padding:0.3rem 0.2rem;border-bottom:1px solid rgba(0,245,255,0.08);">
            <span style="color:rgba(0,245,255,0.4);font-family:'Share Tech Mono',monospace;font-size:0.7rem;margin-right:0.5rem;">{i+1:02d}</span>{cls}
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="footer">ChemClassify AI · 1D-CNN · TensorFlow · Streamlit · Built for Chemistry NLP</div>', unsafe_allow_html=True)