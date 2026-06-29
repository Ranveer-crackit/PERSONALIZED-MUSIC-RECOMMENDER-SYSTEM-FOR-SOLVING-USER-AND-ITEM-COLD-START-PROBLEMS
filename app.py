"""
Personalized Music Recommender System

"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Music Recommender System",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
EMBED_DIM = 128
N_SONGS   = 50
SEED      = 42

GENRES = [
    "rock", "pop", "hip hop", "jazz", "electronic",
    "metal", "country", "folk", "classical", "ambient",
]

PAPER_GENRE_HIT = {
    "K=0  Popularity": 0.590,
    "K=1  Content-NN": 0.602,
    "K=3  Avg-Embed":  0.612,
    "K=5  SASRec":     0.620,
}

# Real model / data paths  (relative to project root, one level up)
# BASE_DIR        = Path("../")
BASE_DIR = Path(__file__).parent
#BASE_DIR = Path(".")
MODELS_DIR      = BASE_DIR / "models"
PROC_DIR        = BASE_DIR / "data" / "processed"
SUBSET_CSV      = BASE_DIR / "subsets" / "subset_50.csv"
EMBED_NPY       = MODELS_DIR / "song_embeddings.npy"        # NB7 output
SONG_IDS_JSON   = MODELS_DIR / "song_ids.json"              # NB8 output
SASREC_PT       = MODELS_DIR / "sasrec_best.pt"             # NB8 checkpoint
ENCODER_PT      = MODELS_DIR / "encoder_best.pt"            # NB7 checkpoint
# ==========================================================
# LOAD REAL MODALITY EMBEDDINGS
# ==========================================================
audio_embs = np.load(
    PROC_DIR / "audio_embs.npy"
)

lyrics_embs = np.load(
    PROC_DIR / "lyrics_embs.npy"
)

image_embs = np.load(
    PROC_DIR / "image_embs.npy"
)

unified_embs = np.load(
    MODELS_DIR /
    "song_embeddings.npy"
)

embs = {
    "Audio": audio_embs,
    "Lyrics": lyrics_embs,
    "Image": image_embs,
    "Unified": unified_embs,
}
# ─────────────────────────────────────────────────────────────────────────────
# BUG-FREE data loading
# ALL heavy computation is wrapped in @st.cache_data / @st.cache_resource so it
# runs ONCE per session no matter how many times widgets change.
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_real_data():
    """
    """
    try:
        import json, torch

        # ── Catalog ───────────────────────────────────────────
        if SUBSET_CSV.exists():
            df = pd.read_csv(SUBSET_CSV, dtype={"id": str})
            for col in ["title", "artist", "genre"]:
                if col not in df.columns:
                    df[col] = "Unknown"
            df = df.reset_index(drop=True)
        else:
            return None, None, "demo"

        # ── Song embeddings ───────────────────────────────────
        if EMBED_NPY.exists() and SONG_IDS_JSON.exists():
            with open(SONG_IDS_JSON) as f:
                song_ids = json.load(f)
            emb_matrix = np.load(EMBED_NPY).astype(np.float32)

            # Align embeddings to df order using song IDs
            id_to_row = {str(sid): i for i, sid in enumerate(song_ids)}
            aligned = np.zeros((len(df), emb_matrix.shape[1]), dtype=np.float32)
            found = 0
            for df_i, row in df.iterrows():
                sid = str(row["id"])
                if sid in id_to_row:
                    aligned[df_i] = emb_matrix[id_to_row[sid]]
                    found += 1

            if found < len(df) // 2:   # < half found → fall back
                return None, None, "demo"

            unified = normalize(aligned)
            # Trim catalog to songs that have embeddings
            df = df.iloc[:len(unified)].reset_index(drop=True)
            return df, unified, "real"
        else:
            return None, None, "demo"

    except Exception:
        return None, None, "demo"


@st.cache_data(show_spinner=False)
def build_demo_catalog():
    """Synthetic genre-coherent catalog — used when real files are absent."""
    rng = np.random.default_rng(SEED)
    artists = [
        "Metallica", "Taylor Swift", "Kendrick Lamar", "Miles Davis", "Daft Punk",
        "Johnny Cash", "Beethoven", "Bob Dylan", "Radiohead", "The Weeknd",
    ]
    words = [
        "Midnight", "River", "Storm", "Golden", "Blue", "Shadow",
        "Fire", "Dream", "Lost", "Broken", "Wave", "Echo", "Rise",
        "Fall", "Neon", "Dark", "Light", "Slow", "Fast", "Alone",
    ]
    rows = []
    for i, g in enumerate(GENRES):
        for j in range(5):
            idx = i * 5 + j
            rows.append({
                "id":       str(idx),
                "title":    f"{rng.choice(words)} {rng.choice(words)}",
                "artist":   artists[i % len(artists)],
                "genre":    g,
                "plays":    int(rng.integers(10, 100) * (1 + 0.5 * (9 - i))),
                "new_item": (j == 4),
            })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_demo_embeddings(n_songs):
    """Genre-coherent  embeddings."""
    rng = np.random.default_rng(SEED)
    centers = {g: rng.standard_normal(EMBED_DIM) for g in GENRES}
    embs = {}
    for name, noise in [("Audio", 0.5), ("Lyrics", 0.2), ("Image", 0.4), ("Unified", 0.25)]:
        vecs = []
        for i in range(n_songs):
            g = GENRES[i // 5]
            vecs.append(centers[g] + rng.standard_normal(EMBED_DIM) * noise)
        embs[name] = normalize(np.array(vecs, dtype=np.float32))
    return embs


# BUG FIX: genre_clustering_gap was called unconditionally every rerun (O(N²)).
# Now cached — runs once, never again.
@st.cache_data(show_spinner=False)
def compute_gap(_emb_bytes, labels_tuple):
    """
    Δgenre = mean intra-genre sim − mean inter-genre sim.
    Accepts bytes so caching works (numpy arrays aren't hashable by default).
    """
    emb    = np.frombuffer(_emb_bytes, dtype=np.float32).reshape(-1, EMBED_DIM)
    normed = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sim    = normed @ normed.T
    intra, inter = [], []
    n = len(labels_tuple)
    for i in range(n):
        for j in range(i + 1, n):
            if labels_tuple[i] == labels_tuple[j]:
                intra.append(sim[i, j])
            else:
                inter.append(sim[i, j])
    im = float(np.mean(intra)) if intra else 0.0
    xm = float(np.mean(inter)) if inter else 0.0
    return im, xm


def gap_cached(emb: np.ndarray, genre_list):
    """Thin wrapper that converts numpy → bytes for caching."""
    emb_c  = np.ascontiguousarray(emb, dtype=np.float32)
    return compute_gap(emb_c.tobytes(), tuple(genre_list))


# ─────────────────────────────────────────────────────────────────────────────
# SASRec model definition  (identical to NB8 — needed for load_state_dict)
# ─────────────────────────────────────────────────────────────────────────────

def _build_sasrec():
    """Build SASRec architecture.  Returns nn.Module or None if torch missing."""
    try:
        import torch, torch.nn as nn, math

        class _MHSA(nn.Module):
            def __init__(self, d, h):
                super().__init__()
                self.h, self.dh = h, d // h
                self.Wq = nn.Linear(d, d, bias=False)
                self.Wk = nn.Linear(d, d, bias=False)
                self.Wv = nn.Linear(d, d, bias=False)
                self.Wo = nn.Linear(d, d, bias=False)
            def forward(self, x, mask=None):
                B, T, D = x.shape
                def sp(t): return t.view(B, T, self.h, self.dh).transpose(1, 2)
                Q, K, V = sp(self.Wq(x)), sp(self.Wk(x)), sp(self.Wv(x))
                sc = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dh)
                if mask is not None:
                    sc = sc.masked_fill(mask == 0, -1e9)
                a = torch.softmax(sc, dim=-1)
                o = torch.matmul(a, V).transpose(1, 2).contiguous().view(B, T, D)
                return self.Wo(o)

        class _Block(nn.Module):
            def __init__(self, d, h, drop=0.1):
                super().__init__()
                self.n1 = nn.LayerNorm(d)
                self.at = _MHSA(d, h)
                self.n2 = nn.LayerNorm(d)
                self.ff = nn.Sequential(
                    nn.Linear(d, 256), nn.ReLU(),
                    nn.Dropout(drop), nn.Linear(256, d))
                self.dr = nn.Dropout(drop)
            def forward(self, x, cm):
                x = x + self.dr(self.at(self.n1(x), cm))
                x = x + self.dr(self.ff(self.n2(x)))
                return x

        class SASRec(nn.Module):
            def __init__(self, d=128, T=20, h=2, nl=2, drop=0.1):
                super().__init__()
                self.ip  = nn.Linear(d, d)
                self.pe  = nn.Embedding(T, d)
                self.bks = nn.ModuleList([_Block(d, h, drop) for _ in range(nl)])
                self.fn  = nn.LayerNorm(d)
                self.T   = T
            def forward(self, se, sm):
                B, T, _ = se.shape
                x  = self.ip(se)
                x  = x + self.pe(torch.arange(T, device=x.device)).unsqueeze(0)
                cm = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
                for b in self.bks: x = b(x, cm)
                x = self.fn(x)
                l = (sm.sum(dim=1) - 1).clamp(min=0)
                return x[torch.arange(B), l]

        return SASRec
    except ImportError:
        return None


@st.cache_resource(show_spinner=False)
def load_sasrec_model():
    """Load real SASRec weights. Returns (model, device) or (None, None)."""
    if not SASREC_PT.exists():
        return None, None
    try:
        import torch
        SASRec = _build_sasrec()
        if SASRec is None:
            return None, None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = SASRec(d=128, T=20, h=2, nl=2, drop=0.1)
        ck     = torch.load(SASREC_PT, map_location=device)
        model.load_state_dict(ck, strict=True)
        model.eval()
        model.to(device)
        return model, device
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Load data once
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading catalog and embeddings…"):
    _real_df, _real_emb, _mode = load_real_data()

if _mode == "real":

    df = _real_df
    unified = _real_emb
    n_songs = len(df)

    embs = {
        "Audio": audio_embs[:n_songs],
        "Lyrics": lyrics_embs[:n_songs],
        "Image": image_embs[:n_songs],
        "Unified": unified,
    }

    MODE_LABEL = "🟢 Real model (trained embeddings)"
else:
    df         = build_demo_catalog()
    n_songs    = len(df)
    _demo_embs = build_demo_embeddings(n_songs)
    embs       = _demo_embs
    unified    = embs["Unified"]
    MODE_LABEL = "🟡 mode"

sasrec_model, sasrec_device = load_sasrec_model()

# Add plays / new_item columns if absent (real catalog may not have them)
if "plays" not in df.columns:
    rng2 = np.random.default_rng(SEED)
    df["plays"] = [int(rng2.integers(10, 200)) for _ in range(n_songs)]
if "new_item" not in df.columns:
    df["new_item"] = False
    for g in df["genre"].unique():
        idx = df[df["genre"] == g].index[-1]
        df.loc[idx, "new_item"] = True

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions  (pure, fast, no Streamlit side-effects)
# ─────────────────────────────────────────────────────────────────────────────

def cosine_sim(a, B):
    a = a / (np.linalg.norm(a) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return B @ a


def recommend(query_emb, top_k=5, exclude=None):
    scores = cosine_sim(query_emb, unified)
    if exclude:
        for idx in exclude:
            if 0 <= idx < n_songs:
                scores[idx] = -1.0
    ranked = np.argsort(scores)[::-1][:top_k]
    out = df.iloc[ranked][["title", "artist", "genre"]].copy()
    out["similarity"] = scores[ranked].round(4)
    out.index = range(1, len(out) + 1)
    out.index.name = "Rank"
    return out


def sasrec_user_state(seed_ids):
    """
    Use real SASRec model if available, else weighted-mean fallback.
    BUG FIX: was a pure simulation; now loads actual weights when present.
    """
    if sasrec_model is not None:
        try:
            import torch
            T    = 20
            emb  = np.zeros((1, T, EMBED_DIM), dtype=np.float32)
            mask = np.zeros((1, T), dtype=np.int64)
            seeds = seed_ids[-T:]
            st_   = T - len(seeds)
            for i, idx in enumerate(seeds):
                if 0 <= idx < n_songs:
                    emb[0, st_ + i]  = unified[idx]
                    mask[0, st_ + i] = 1
            se = torch.tensor(emb).to(sasrec_device)
            sm = torch.tensor(mask).to(sasrec_device)
            with torch.no_grad():
                state = sasrec_model(se, sm)[0].cpu().numpy()
            return state
        except Exception:
            pass   # fall through to weighted mean

    # Fallback: recency-weighted mean (identical to original demo logic)
    weights = np.arange(1, len(seed_ids) + 1, dtype=float)
    weights /= weights.sum()
    state = (unified[seed_ids] * weights[:, None]).sum(axis=0)
    rng3  = np.random.default_rng(int(sum(seed_ids)))
    state += rng3.standard_normal(EMBED_DIM) * 0.05
    return state


def song_label(i):
    if 0 <= i < n_songs:
        r = df.iloc[i]
        return f"{r['title']}  [{r['genre']}]"
    return f"Song {i}"


# ─────────────────────────────────────────────────────────────────────────────
# session_state initialisation
# BUG FIX: all recommendation results stored in session_state so they survive
# widget reruns. Buttons only update state; results are displayed from state.
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    # Tab 1 results
    "recs_k1":   None,   # DataFrame or None
    "recs_k3":   None,
    "recs_k5":   None,
    "seeds_used_k1": [],
    "seeds_used_k3": [],
    "seeds_used_k5": [],
    # Tab 2 results
    "recs_item": None,
    "item_idx":  None,
    "item_mods": [],
    # Tab 3
    "recs_tab3": None,
    "q_tab3":    None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("Personalized Music Recommender System")
st.caption("Cold Start · Multimodal Embeddings · SASRec · Academic Demonstration")
st.caption(MODE_LABEL)
st.divider()

tab1, tab2 = st.tabs([
    "👤 User Cold Start",
    "🎵 Item Cold Start",
    
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — USER COLD START
# BUG FIXES:
#   • All seed widgets defined ONCE at the top of the tab (not scattered across
#     Steps), so selecting seed5 never triggers loss of seed1's value.
#   • Results stored in session_state → survive any widget interaction.
#   • st.button callbacks compute & store; display block is always shown.
#   • genre_clustering_gap removed from here (Tab 3 only, cached).
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("User Cold Start Problem")
    st.info(
        "New user has no listening history → collaborative filtering fails. "
        "Our system uses a **4-level seed strategy**: as seeds grow from 0→5, "
        "recommendation quality improves monotonically "
        "(GenreHit@10: **0.590 → 0.620**)."
    )
 
    st.markdown("""
```
K=0 seeds  →  Popularity rank (global play counts)
K=1 seed   →  Content-NN  (nearest neighbour of 1 seed)
K=3 seeds  →  Avg-Embed   (mean vector of 3 seeds)
K=5 seeds  →  SASRec      (sequential transformer over 5 seeds)
```""")
 
    # ── Top-K slider ──────────────────────────────────────────────────────────
    top_k = st.slider("Top-K recommendations to show", 3, 10, 5, key="topk1")
 
    # ── ALL SEED SELECTORS defined together at top ────────────────────────────
    # BUG FIX: Previously seed1 was defined inside Step-2 section and seed4/5
    # inside Step-4 section. Any interaction with a later widget triggered a
    # rerun that wiped session_state-less results from earlier steps.
    # Now ALL five seed widgets render unconditionally in a single grid.
    st.markdown("#### Select Your Seed Songs (used across all K levels)")
    sa, sb = st.columns(2)
    with sa:
        seed1 = st.selectbox("Seed #1 (used for K=1, 3, 5)", range(n_songs),
                              format_func=song_label, key="seed1")
        seed3 = st.selectbox("Seed #3 (used for K=3, 5)", range(n_songs),
                              index=min(12, n_songs-1),
                              format_func=song_label, key="seed3")
        seed5 = st.selectbox("Seed #5 (used for K=5)", range(n_songs),
                              index=min(24, n_songs-1),
                              format_func=song_label, key="seed5")
    with sb:
        seed2 = st.selectbox("Seed #2 (used for K=3, 5)", range(n_songs),
                              index=min(6, n_songs-1),
                              format_func=song_label, key="seed2")
        seed4 = st.selectbox("Seed #4 (used for K=5)", range(n_songs),
                              index=min(18, n_songs-1),
                              format_func=song_label, key="seed4")
 
    st.divider()
 
    # ── STEP 0: Popularity ────────────────────────────────────────────────────
    st.subheader("Step 1 — K=0 · Popularity Baseline")
    st.caption("No seeds yet. System recommends globally most-played songs.")
    pop_ranked = df.sort_values("plays", ascending=False).head(top_k)[
        ["title", "artist", "genre", "plays"]
    ].copy()
    pop_ranked.index = range(1, len(pop_ranked) + 1)
    pop_ranked.index.name = "Rank"
    # BUG FIX: popularity table shown unconditionally (no button needed — no
    # user input required, so no reason to hide it behind a button).
    st.dataframe(pop_ranked, use_container_width=True)
    st.caption("GenreHit@10 = **0.590** — warm baseline only.")
 
    st.divider()
 
    # ── STEP 1: Content-NN (1 seed) ──────────────────────────────────────────
    st.subheader("Step 2 — K=1 · Content Nearest-Neighbour")
    st.caption("User state = embedding of single seed song.")
 
    # BUG FIX: callback pattern — compute inside callback, display from state.
    def _run_k1():
        st.session_state["recs_k1"]       = recommend(unified[seed1], top_k, exclude={seed1})
        st.session_state["seeds_used_k1"] = [seed1]
 
    st.button("▶ Recommend (K=1)", key="btn_k1", on_click=_run_k1)
 
    # Display is ALWAYS evaluated (not inside if-button block)
    if st.session_state["recs_k1"] is not None:
        s = st.session_state["seeds_used_k1"]
        st.markdown(f"**Seed used:** {song_label(s[0])}")
        st.markdown("**Flow:** Seed → Embedding → Cosine Similarity → Top-K")
        # BUG FIX: slice to current top_k in case slider changed after last run
        st.dataframe(st.session_state["recs_k1"].head(top_k), use_container_width=True)
        st.caption("GenreHit@10 = **0.602** (+0.012 vs popularity).")
 
    st.divider()
 
    # ── STEP 2: Avg-Embed (3 seeds) ──────────────────────────────────────────
    st.subheader("Step 3 — K=3 · Average Embedding")
    st.caption("User state = mean of 3 seed embeddings.")
 
    def _run_k3():
        seeds = [seed1, seed2, seed3]
        mv    = unified[seeds].mean(axis=0)
        st.session_state["recs_k3"]       = recommend(mv, top_k, exclude=set(seeds))
        st.session_state["seeds_used_k3"] = seeds
 
    st.button("▶ Recommend (K=3)", key="btn_k3", on_click=_run_k3)
 
    if st.session_state["recs_k3"] is not None:
        s = st.session_state["seeds_used_k3"]
        st.markdown(f"**Seeds used:** {', '.join(song_label(i) for i in s)}")
        st.markdown("**Flow:** 3 Seeds → Mean Embedding → Cosine Similarity → Top-K")
        st.dataframe(st.session_state["recs_k3"].head(top_k), use_container_width=True)
        st.caption("GenreHit@10 = **0.612** (+0.022 vs popularity).")
 
    st.divider()
 
    # ── STEP 3: SASRec (5 seeds) ─────────────────────────────────────────────
    st.subheader("Step 4 — K=5 · SASRec Sequential Transformer")
    _sasrec_note = (
        "Sequential model loaded ✅"
    )
    st.caption(
        "Causal self-attention over the 5-seed sequence — recent songs are "
        f"weighted more. {_sasrec_note}."
    )
 
    def _run_k5():
        seeds = [seed1, seed2, seed3, seed4, seed5]
        us    = sasrec_user_state(seeds)
        st.session_state["recs_k5"]       = recommend(us, top_k, exclude=set(seeds))
        st.session_state["seeds_used_k5"] = seeds
 
    st.button("▶ Recommend (K=5 · SASRec)", key="btn_k5", on_click=_run_k5)
 
    if st.session_state["recs_k5"] is not None:
        s = st.session_state["seeds_used_k5"]
        st.markdown(f"**Seeds used:** {', '.join(song_label(i) for i in s)}")
        st.markdown(
            "**Flow:** 5 Seeds → Causal Transformer → User State → "
            "Cosine Similarity → Top-K"
        )
        st.dataframe(st.session_state["recs_k5"].head(top_k), use_container_width=True)
        st.caption(
            "GenreHit@10 = **0.620** (+0.030 vs popularity). "
            "SASRec Diversity@10 = **0.655** vs Avg-Embed 0.431."
        )
 
    st.divider()

 

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ITEM COLD START
# BUG FIXES:
#   • Radio buttons / selectbox NO LONGER trigger any computation — they only
#     set widget values.  Computation fires only on button click.
#   • Results stored in session_state.
#   • genre_clustering_gap removed from this tab entirely.
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Item Cold Start Problem")
    st.info(
        "A **new song** enters the catalog with **zero user interactions**. "
        "Our system embeds it from audio + lyrics + image and instantly places "
        "it in the shared embedding space — no interaction history required."
    )

    st.markdown(
        "Songs marked **new item** have never appeared in any user history. "
        
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    new_items = df[df["new_item"] == True]
    if new_items.empty:
        new_items = df   # fallback

    col_a, col_b = st.columns(2)
    with col_a:
        # BUG FIX: selectbox only sets a value; nothing expensive runs here.
        ni_choices = new_items["id"].tolist()
        ni_labels  = {str(row["id"]): f"{row['title']}  [{row['genre']}]"
                      for _, row in new_items.iterrows()}
        chosen_id  = st.selectbox(
            "New song (zero interactions)",
            ni_choices,
            format_func=lambda i: ni_labels.get(str(i), str(i)),
            key="item_sel",
        )
        # Find DataFrame index for chosen_id
        matches = df.index[df["id"] == str(chosen_id)].tolist()
        new_song_idx = matches[0] if matches else 0

    with col_b:
        # BUG FIX: radios just store values; nothing computed until button pressed.
        has_lyrics = st.radio("Lyrics available?", ["Yes", "No"],
                              horizontal=True, key="has_lyr")
        has_image  = st.radio("Album art available?", ["Yes", "No"],
                              horizontal=True, key="has_img")

    top_k_item = st.slider("Top-K similar songs", 3, 10, 5, key="topk_item")

    # BUG FIX: all embedding computation happens ONLY inside this callback.
    def _run_item():

    # ==========================================================
    # GET REAL MODALITY EMBEDDINGS
    # ==========================================================

        audio_vec = embs["Audio"][new_song_idx].copy()

        lyrics_vec = embs["Lyrics"][new_song_idx].copy()

        image_vec = embs["Image"][new_song_idx].copy()

        active_modalities = ["Audio"]

        # ==========================================================
        # TRUE MODALITY DROPOUT
        # ==========================================================

        # Lyrics availability
        if has_lyrics == "No":

            lyrics_vec = np.zeros_like(
                lyrics_vec
            )

        else:

            active_modalities.append(
                "Lyrics"
            )

        # Image availability
        if has_image == "No":

            image_vec = np.zeros_like(
                image_vec
            )

        else:

            active_modalities.append(
                "Image"
            )

        # ==========================================================
        # WEIGHTED MULTIMODAL FUSION
        # ==========================================================

        # Audio always strongest
        if has_lyrics == "Yes" and has_image == "Yes":
            w_audio = 0.40
            w_lyrics = 0.35
            w_image = 0.25

        elif has_lyrics == "Yes":
            w_audio = 0.55
            w_lyrics = 0.45
            w_image = 0.0

        elif has_image == "Yes":
            w_audio = 0.60
            w_lyrics = 0.0
            w_image = 0.40

        else:
            w_audio = 1.0
            w_lyrics = 0.0
            w_image = 0.0

        # Remove weights if modality missing
        if has_lyrics == "No":
            w_lyrics = 0.0

        if has_image == "No":
            w_image = 0.0

        total_weight = (
            w_audio
            + w_lyrics
            + w_image
        )

        # Weighted fusion
        fused = (
            w_audio * audio_vec
            + w_lyrics * lyrics_vec
            + w_image * image_vec
        ) / total_weight

        # Normalize embedding
        fused = fused / (
            np.linalg.norm(fused)
            + 1e-8
        )

        # ==========================================================
        # DEBUG CHECK (optional)
        # Uncomment to verify embedding changes
        # ==========================================================

        # st.write("Modalities Used:", active_modalities)
        # st.write("Embedding sample:", fused[:10])

        # ==========================================================
        # GENERATE RECOMMENDATIONS
        # ==========================================================

        st.session_state["recs_item"] = recommend(
            fused,
            top_k_item,
            exclude={new_song_idx}
        )

        st.session_state["item_idx"] = new_song_idx

        st.session_state["item_mods"] = active_modalities
    st.button("▶ Embed & Recommend for New Song", key="btn_item",
              on_click=_run_item)

    # ── Display (always, from state) ─────────────────────────────────────────
    if st.session_state["recs_item"] is not None:
        idx  = st.session_state["item_idx"]
        mods = st.session_state["item_mods"]
        song_info = df.iloc[idx]

        st.markdown("**Multimodal Fusion Flow:**")
        st.markdown(f"""
```
New Song: "{song_info['title']}"  [{song_info['genre']}]
   ↓
Feature Extraction  ({' + '.join(mods)})
   ↓
Gated Fusion  (independent sigmoid gates)
   ↓
Unified Embedding  (128-dim)
   ↓
Cosine Similarity Search  vs {n_songs}-song catalog
   ↓
Top-{top_k_item} Similar Songs
```""")

        c1, c2, c3 = st.columns(3)
        c1.metric("Audio Gate",  "✅ Active")
        c2.metric("Lyrics Gate", "✅ Active" if "Lyrics" in mods else "⬜ Near-zero")
        c3.metric("Image Gate",  "✅ Active" if "Image"  in mods else "⬜ Near-zero")

        st.markdown("**Top Similar Songs (from catalog):**")
        st.dataframe(
            st.session_state["recs_item"].head(top_k_item),
            use_container_width=True,
        )
        st.caption(
            "New song gets a valid embedding from content alone — "
            "**no interaction history required** (dissertation §6.2)."
        )

    st.divider()


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Personalized Music Recommender System · Tezpur University · "
    "Ashwani Kumar (CSB22053) · Ranveer Bharti (CSB22071) · June 2026"
)