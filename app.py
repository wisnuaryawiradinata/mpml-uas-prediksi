import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ---------- App config ----------
st.set_page_config(
    page_title="Prediksi Profitabilitas Menu Restoran",
    page_icon="🍽️",
    layout="centered"
)

st.title("Prediksi Profitabilitas Menu Restoran")

# ---------- Load model (cached) ----------
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        return pickle.load(f)

pipeline = load_model()

# ---------- Sidebar: info & preset ----------
with st.sidebar:
    st.header("ℹ️ Petunjuk")
    st.write(
        "Isi **Harga** dan **Jumlah bahan** lalu klik *Prediksi*. "
        "Model: Random Forest (pipeline + scaler)."
    )
    st.divider()
    st.subheader("🎛 Preset contoh")
    preset = st.selectbox(
        "Pilih contoh input",
        ["— (manual) —", "Murah & bahan sedikit", "Sedang", "Mahal & bahan banyak"]
    )
    if preset != "— (manual) —":
        if preset == "Murah & bahan sedikit":
            st.session_state["price"] = 6.0
            st.session_state["ingredient_count"] = 3
        elif preset == "Sedang":
            st.session_state["price"] = 11.0
            st.session_state["ingredient_count"] = 7
        else:  # Mahal & bahan banyak
            st.session_state["price"] = 20.0
            st.session_state["ingredient_count"] = 15
        st.success("Preset diterapkan. Lihat nilai di form utama.")

# ---------- Inputs ----------
price = st.number_input(
    "Harga menu ($)", min_value=0.0, max_value=100.0, step=0.5,
    value=st.session_state.get("price", 10.0)
)
ingredient_count = st.number_input(
    "Jumlah bahan yang digunakan", min_value=1, max_value=20, step=1,
    value=st.session_state.get("ingredient_count", 5)
)

col1, col2 = st.columns([1,1])
with col1:
    predict_clicked = st.button("Prediksi", type="primary")
with col2:
    if st.button("Reset"):
        st.session_state.clear()
        st.rerun()

# ---------- Predict ----------
def make_pred(price_val: float, ing_count_val: int):
    df = pd.DataFrame(
        {"Price": [price_val], "IngredientCount": [ing_count_val]}
    )
    # Prediksi label
    y_pred = pipeline.predict(df)[0]
    # Probabilities (jika tersedia)
    y_proba = None
    if hasattr(pipeline, "predict_proba") or hasattr(pipeline.named_steps.get("model", None), "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(df)[0]
        except Exception:
            y_proba = None
    return y_pred, y_proba

label_map = {0: "Low", 1: "Medium", 2: "High"}

if predict_clicked:
    # Validasi simple
    if price <= 0:
        st.error("Harga harus lebih besar dari 0.")
    else:
        pred, proba = make_pred(price, ingredient_count)
        st.success(f"Prediksi profitabilitas: **{label_map.get(pred, pred)}**")

        # tampilkan probabilitas bila ada
        if proba is not None:
            prob_df = pd.DataFrame(
                {
                    "Kelas": ["Low", "Medium", "High"],
                    "Probabilitas": np.round(proba, 4)
                }
            )
            st.caption("Detail probabilitas prediksi:")
            st.dataframe(prob_df, use_container_width=True)

# ---------- Footer ----------
st.caption(
    "Model: Random Forest (Pipeline dengan StandardScaler). "
    "Input yang masuk akal: Harga 0–100, Bahan 1–20."
)
