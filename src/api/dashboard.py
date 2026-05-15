"""
dashboard.py — Streamlit dashboard for Running Form Analyzer.
Run: streamlit run src/api/dashboard.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = "http://localhost:8000"

CLASS_COLORS = {
    "good_form":    "#2ecc71",
    "overstriding": "#e74c3c",
    "forward_lean": "#f39c12",
    "arm_crossing": "#9b59b6",
}
CLASS_ICONS = {
    "good_form":    "✅",
    "overstriding": "⚠️",
    "forward_lean": "⚠️",
    "arm_crossing": "⚠️",
}
CLASS_DESCRIPTIONS = {
    "good_form":    "Efficient mechanics — upright posture, symmetric arm swing.",
    "overstriding": "Foot landing ahead of center of mass — braking force, injury risk.",
    "forward_lean": "Excessive trunk flexion — suggests fatigue or weak core.",
    "arm_crossing": "Arms crossing body midline — energy waste, rotational inefficiency.",
}

st.set_page_config(page_title="Running Form Analyzer", page_icon="🏃", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏃 Running Form Analyzer")
    st.markdown("Upload a running clip to get AI-powered form analysis and coaching feedback.")
    api_url = st.text_input("API URL", value=API_URL)
    st.divider()
    try:
        r = requests.get(f"{api_url}/health", timeout=2)
        h = r.json()
        icon = "🟢" if h["status"] == "ok" else "🟡"
        st.markdown(f"{icon} **API:** {h['status'].upper()}")
        st.markdown(f"{'✅' if h['classifier_loaded'] else '❌'} BiLSTM Classifier")
        st.markdown(f"{'✅' if h['scorer_loaded'] else '❌'} Form Scorer")
    except Exception:
        st.warning("API not reachable.\n`uvicorn src.api.main:app`")
    st.divider()
    st.markdown("**Running Form Classes**")
    for cls, desc in CLASS_DESCRIPTIONS.items():
        st.markdown(f"{CLASS_ICONS[cls]} **{cls.replace('_',' ').title()}**")
        st.caption(desc)
    st.divider()
    st.markdown("**Ideal Benchmarks**")
    for k, v in {
        "Trunk lean":    "3–12°",
        "Overstride":    "foot ≤ 0.05× torso ahead",
        "Arm symmetry":  "< 15° difference",
        "Hip drop":      "< 5°",
        "Knee drive":    "> 60°",
        "Vert. osc.":    "< 0.08 torso units",
    }.items():
        st.markdown(f"- **{k}**: {v}")

# ── Main ─────────────────────────────────────────────────────────────────────
st.title("🏃 Running Form Analysis Dashboard")
st.markdown("*MediaPipe pose estimation → BiLSTM classification → biomechanical feedback*")

tab1, tab2, tab3 = st.tabs(["🎥 Analyze Video", "📊 Feature Explorer", "ℹ️ About"])

# ── Tab 1: Video Analysis ─────────────────────────────────────────────────────
with tab1:
    uploaded = st.file_uploader(
        "Upload a running video clip",
        type=["mp4", "avi", "mov", "mkv"],
        help="Best: side-on or rear view, full body visible, 5–30 seconds",
    )

    if uploaded:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("📹 Video")
            st.video(uploaded)
            st.caption(f"{uploaded.name} | {len(uploaded.getvalue())/1024/1024:.1f} MB")

        with c2:
            st.subheader("🔬 Analysis")
            if st.button("▶ Analyze Running Form", type="primary", use_container_width=True):
                with st.spinner("Extracting pose → computing features → classifying..."):
                    try:
                        resp = requests.post(
                            f"{api_url}/analyze",
                            files={"video": (uploaded.name, uploaded.getvalue(), "video/mp4")},
                            timeout=180,
                        )
                        if resp.status_code == 200:
                            st.session_state["result"] = resp.json()
                        else:
                            st.error(f"API error {resp.status_code}: {resp.text[:300]}")
                    except requests.ConnectionError:
                        st.error("Cannot connect to API.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        # ── Results ───────────────────────────────────────────────────────────
        if "result" in st.session_state:
            result = st.session_state["result"]
            st.divider()

            pred   = result.get("form_classification", {})
            form   = result.get("form_analysis", {})
            cls    = pred.get("form_class", "unknown")
            score  = form.get("form_score")
            conf   = pred.get("confidence")

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Frames", result.get("total_frames", "—"))
            m2.metric("Pose Detected", f"{result.get('detection_rate', 0):.0%}")
            m3.metric("Form Class",    f"{CLASS_ICONS.get(cls, '❓')} {cls.replace('_',' ').title()}")
            if score is not None:
                m4.metric("Form Score", f"{score:.0f}/100")

            # Class description
            desc = CLASS_DESCRIPTIONS.get(cls, "")
            color = CLASS_COLORS.get(cls, "#888")
            st.markdown(
                f"<div style='background:{color}22;border-left:4px solid {color};"
                f"padding:12px 16px;border-radius:6px;margin:8px 0'>"
                f"<b>{CLASS_ICONS.get(cls,'❓')} {cls.replace('_',' ').title()}</b><br>{desc}</div>",
                unsafe_allow_html=True,
            )

            # Form score gauge
            if score is not None:
                col_gauge, col_proba = st.columns([1, 1])
                with col_gauge:
                    st.subheader("Form Quality Score")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=score,
                        title={"text": "Score / 100"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar":  {"color": "#2ecc71" if score >= 75 else "#f39c12" if score >= 50 else "#e74c3c"},
                            "steps": [
                                {"range": [0,  50],  "color": "#fde9e9"},
                                {"range": [50, 75],  "color": "#fef9e7"},
                                {"range": [75, 100], "color": "#e9f7ef"},
                            ],
                            "threshold": {"line": {"color": "black", "width": 3},
                                          "thickness": 0.75, "value": 75},
                        }
                    ))
                    fig.update_layout(height=280)
                    st.plotly_chart(fig, use_container_width=True)

                with col_proba:
                    st.subheader("Class Probabilities")
                    probs = pred.get("probabilities", {})
                    if probs:
                        prob_df = pd.DataFrame(
                            sorted(probs.items(), key=lambda x: x[1], reverse=True),
                            columns=["Class", "Probability"]
                        )
                        prob_df["Class"] = prob_df["Class"].str.replace("_", " ").str.title()
                        fig2 = px.bar(
                            prob_df, x="Probability", y="Class",
                            orientation="h", range_x=[0, 1],
                            color="Probability",
                            color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
                        )
                        fig2.update_layout(height=280, showlegend=False)
                        st.plotly_chart(fig2, use_container_width=True)

            # Attention weights
            attn = pred.get("attention_weights")
            if attn:
                st.subheader("Temporal Attention — Which gait phase triggered this classification?")
                fig_attn = go.Figure()
                fig_attn.add_trace(go.Scatter(
                    x=list(range(len(attn))), y=attn,
                    fill="tozeroy", mode="lines",
                    line={"color": color, "width": 2},
                    fillcolor=f"{color}44",
                ))
                fig_attn.add_vline(x=int(len(attn) * 0.5), line_dash="dot",
                                   annotation_text="Mid-swing")
                fig_attn.update_layout(
                    height=180,
                    xaxis_title="Frame",
                    yaxis_title="Attention Weight",
                    margin={"t": 20, "b": 30},
                )
                st.plotly_chart(fig_attn, use_container_width=True)

            # Biomechanical readings
            feats = form.get("release_features", {})
            if feats:
                st.subheader("📐 Biomechanical Readings")
                feat_items = [(k.replace("_", " ").title(), round(v, 2))
                              for k, v in feats.items() if isinstance(v, (int, float))]
                if feat_items:
                    feat_df = pd.DataFrame(feat_items, columns=["Feature", "Value"])
                    st.dataframe(feat_df, use_container_width=True, hide_index=True)

            # Feedback
            feedback = form.get("feedback", [])
            st.subheader(f"💡 Coaching Feedback ({len(feedback)} item{'s' if len(feedback)!=1 else ''})")
            if feedback:
                for item in feedback:
                    sev   = item.get("severity", "low")
                    icon  = "🔴" if sev == "high" else "🟡"
                    st.markdown(f"{icon} {item['message']}")
            else:
                st.success("✅ No major form faults detected! Great running mechanics.")


# ── Tab 2: Feature Explorer ───────────────────────────────────────────────────
with tab2:
    st.subheader("📊 Feature Explorer")
    st.markdown("Upload `biomech_features.csv` to explore class differences interactively.")

    feat_file = st.file_uploader("Upload biomech_features.csv", type=["csv"], key="feat_exp")
    if feat_file:
        df = pd.read_csv(feat_file)
        if "form_class" in df.columns:
            clip_means = df.groupby(["video_stem", "form_class"]).mean(numeric_only=True).reset_index()
            st.markdown(f"**{clip_means['video_stem'].nunique()} clips** | "
                        f"{clip_means['form_class'].nunique()} classes")

            numeric = [c for c in clip_means.select_dtypes(include=[np.number]).columns
                       if c not in ["frame", "timestamp_ms"]]
            col_x = st.selectbox("X feature", numeric, index=0)
            col_y = st.selectbox("Y feature", numeric, index=min(1, len(numeric)-1))

            fig = px.scatter(
                clip_means, x=col_x, y=col_y, color="form_class",
                color_discrete_map=CLASS_COLORS,
                title=f"{col_x} vs {col_y}",
                hover_data=["video_stem"],
                opacity=0.7,
            )
            st.plotly_chart(fig, use_container_width=True)

            feat_box = st.selectbox("Feature for distribution", numeric)
            fig2 = px.box(
                clip_means, x="form_class", y=feat_box,
                color="form_class", color_discrete_map=CLASS_COLORS,
                title=f"{feat_box} by Form Class",
            )
            st.plotly_chart(fig2, use_container_width=True)


# ── Tab 3: About ──────────────────────────────────────────────────────────────
with tab3:
    st.subheader("About Running Form Analyzer")
    st.markdown("""
    | Stage | Description |
    |---|---|
    | Data Engineering | yt-dlp video scraping, MediaPipe pose extraction, normalization |
    | Feature Engineering | 18 per-frame biomechanical features (trunk lean, overstride, etc.) |
    | Stride Analysis | Cadence, vertical oscillation, ground contact ratio |
    | Classification | BiLSTM + Attention (4 running form classes) |
    | Form Scoring | XGBoost regressor (0–100 quality score) |
    | Feedback | Rule-based coaching engine with 9 biomechanical rules |
    | API | FastAPI REST endpoint |
    | Dashboard | This Streamlit app |
    | MLOps | MLflow + Prometheus + Grafana |

    **GitHub**: [github.com/yourusername/running-form-analyzer](https://github.com)
    """)
