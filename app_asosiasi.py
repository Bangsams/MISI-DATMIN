import streamlit as st
import pandas as pd
import numpy as np
import os
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DataMart | Toko Data Mining",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0e1a !important;
    font-family: 'Sora', sans-serif !important;
    color: #e8eaf0 !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: #0d1120 !important; border-right: 1px solid #1e2540; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Tabs */
[data-testid="stTabs"] > div:first-child { border-bottom: 2px solid #1e2540; gap: 0; }
[data-testid="stTabs"] button {
    font-family: 'Sora', sans-serif !important; font-weight: 600 !important;
    font-size: 0.85rem !important; color: #6b7699 !important;
    padding: 0.75rem 1.5rem !important; border-radius: 0 !important;
    border-bottom: 2px solid transparent !important; margin-bottom: -2px !important;
}
[data-testid="stTabs"] button:hover { color: #a78bfa !important; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #a78bfa !important; border-bottom: 2px solid #a78bfa !important;
    background: transparent !important;
}

/* Buttons — default purple */
div.stButton > button {
    font-family: 'Sora', sans-serif !important; font-weight: 600 !important;
    font-size: 0.8rem !important;
    background: linear-gradient(135deg, #7c3aed, #a78bfa) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    padding: 0.5rem 1rem !important; cursor: pointer !important;
    transition: all 0.2s !important; width: 100% !important;
}
div.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(167,139,250,0.4) !important;
}

/* Quantity ± buttons — compact, pastikan tidak hilang */
div[data-testid="column"] div.stButton > button[kind="secondary"],
div[data-testid="column"] div.stButton > button {
    font-size: 1rem !important;
    padding: 0.25rem 0 !important;
    min-width: 2rem !important;
    width: 100% !important;
    min-height: 2rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    line-height: 1 !important;
}

/* Inputs */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] input,
input[type="password"] {
    background: #111827 !important; border: 1px solid #1e2540 !important;
    border-radius: 8px !important; color: #e8eaf0 !important;
    font-family: 'Sora', sans-serif !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #111827 !important; border: 1px solid #1e2540 !important;
    border-radius: 12px !important; padding: 1rem !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.6rem !important; color: #a78bfa !important;
}
[data-testid="stMetricLabel"] { color: #6b7699 !important; font-size: 0.8rem !important; }
[data-testid="stDataFrame"] { border: 1px solid #1e2540 !important; border-radius: 10px !important; overflow: hidden !important; }
hr { border-color: #1e2540 !important; }
[data-testid="stAlert"] { border-radius: 10px !important; font-family: 'Sora', sans-serif !important; }
[data-testid="stExpander"] { background: #111827 !important; border: 1px solid #1e2540 !important; border-radius: 10px !important; }
[data-testid="stCaptionContainer"] { color: #6b7699 !important; font-size: 0.75rem !important; }

/* Dialog / modal */
[data-testid="stDialog"] > div {
    background: #111827 !important; border: 1px solid #1e2540 !important;
    border-radius: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
ADMIN_USER = "datmin"
ADMIN_PASS = "datmin123"

# ─────────────────────────────────────────────
# GOOGLE SHEETS HELPERS
# ─────────────────────────────────────────────
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

@st.cache_resource
def get_worksheet():
    """Koneksi ke Google Sheet — di-cache agar tidak reconnect tiap rerun."""
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=SCOPES
    )
    client = gspread.authorize(creds)
    sh     = client.open(st.secrets["google_sheets"]["sheet_name"])
    ws     = sh.sheet1
    # Buat header jika sheet masih kosong
    if ws.row_count == 0 or ws.cell(1, 1).value != "transaksi_id":
        ws.clear()
        ws.append_row(["transaksi_id", "tanggal"] + ["Kopi","Gula","Susu","Roti","Mentega"])
    return ws

def load_sales() -> pd.DataFrame:
    """Baca semua baris dari Google Sheet sebagai DataFrame."""
    try:
        ws   = get_worksheet()
        data = ws.get_all_records()
        if not data:
            return pd.DataFrame(columns=["transaksi_id", "tanggal"] + ["Kopi","Gula","Susu","Roti","Mentega"])
        df = pd.DataFrame(data)
        for item in ["Kopi","Gula","Susu","Roti","Mentega"]:
            if item in df.columns:
                df[item] = pd.to_numeric(df[item], errors="coerce").fillna(0).astype(int)
        return df
    except Exception as e:
        st.error(f"❌ Gagal membaca Google Sheet: {e}")
        return pd.DataFrame(columns=["transaksi_id", "tanggal"] + ["Kopi","Gula","Susu","Roti","Mentega"])

def save_transaction(keranjang: dict) -> int:
    """Tambah satu baris transaksi baru ke Google Sheet."""
    try:
        ws      = get_worksheet()
        records = ws.get_all_records()
        tid     = len(records) + 1
        tanggal = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row     = [tid, tanggal] + [keranjang.get(item, 0) for item in ["Kopi","Gula","Susu","Roti","Mentega"]]
        ws.append_row(row, value_input_option="USER_ENTERED")
        return tid
    except Exception as e:
        st.error(f"❌ Gagal menyimpan transaksi: {e}")
        return -1

katalog = {
    "Kopi":    {"harga": 25000, "harga_str": "Rp 25.000", "emoji": "☕", "desc": "Kopi Arabika Premium",  "img": "https://images.unsplash.com/photo-1559525839-b184a4d698c7?w=400&q=80"},
    "Gula":    {"harga": 15000, "harga_str": "Rp 15.000", "emoji": "🍬", "desc": "Gula Pasir Murni",      "img": "https://siopen.balangankab.go.id/storage/merchant/products/2025/02/10/f86ef4a74aa4bbc97844ea4390efa05b.jpg"},
    "Susu":    {"harga": 20000, "harga_str": "Rp 20.000", "emoji": "🥛", "desc": "Susu Segar Full Cream", "img": "https://images.unsplash.com/photo-1550583724-b2692b85b150?w=400&q=80"},
    "Roti":    {"harga": 18000, "harga_str": "Rp 18.000", "emoji": "🍞", "desc": "Roti Gandum Utuh",      "img": "https://images.unsplash.com/photo-1598373182133-52452f7691ef?w=400&q=80"},
    "Mentega": {"harga": 12000, "harga_str": "Rp 12.000", "emoji": "🧈", "desc": "Mentega Sapi Asli",     "img": "https://images.unsplash.com/photo-1588195538326-c5b1e9f80a1b?w=400&q=80"},
}
ITEM_NAMES = list(katalog.keys())
COLORS = ["#a78bfa", "#7c3aed", "#34d399", "#fbbf24", "#f87171"]

# ─────────────────────────────────────────────
# APRIORI  (konversi qty → boolean untuk ARM)
# ─────────────────────────────────────────────
def run_apriori(df_sales: pd.DataFrame, min_sup: float, min_conf: float, min_lift: float):
    items = [c for c in ITEM_NAMES if c in df_sales.columns]
    if len(df_sales) < 3:
        return None, None
    # konversi kuantitas ke boolean: qty > 0 → True
    matrix = df_sales[items].gt(0)
    try:
        freq = apriori(matrix, min_support=min_sup, use_colnames=True)
        if freq.empty:
            return None, None
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
        rules = rules[rules["lift"] >= min_lift]
        return freq, rules
    except Exception:
        return None, None

# ─────────────────────────────────────────────
# REKOMENDASI  — selalu return list
# ─────────────────────────────────────────────
def get_recommendations(items_in_cart: list, rules: pd.DataFrame) -> list:
    """
    Ambil rekomendasi berdasarkan semua item di keranjang.
    Selalu mengembalikan list of (item_name, rule_row).
    List kosong ([]) berarti tidak ada rekomendasi.
    """
    if rules is None or rules.empty or not items_in_cart:
        return []
    cart_set = set(items_in_cart)
    # aturan yang antecedent-nya beririsan dengan keranjang
    mask = rules["antecedents"].apply(lambda x: bool(x & cart_set))
    recs = rules[mask].copy()
    if recs.empty:
        return []
    # buang consequent yang sudah ada di keranjang
    recs = recs[recs["consequents"].apply(lambda x: not bool(x & cart_set))]
    if recs.empty:
        return []
    recs = recs.sort_values("confidence", ascending=False)
    # collapse: tiap item rekomendasi hanya muncul sekali (confidence tertinggi)
    seen: dict = {}
    rows: list = []
    for _, rule_row in recs.iterrows():
        for item in rule_row["consequents"]:
            if item not in seen:
                seen[item] = rule_row
                rows.append((item, rule_row))
    return rows  # list of (item_name, rule_Series)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "keranjang": {},        # {item: qty}
    "checkout_done": False,
    "last_tid": None,
    "admin_logged": False,
    "show_popup": False,
    "popup_item": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# POPUP REKOMENDASI  (st.dialog)
# ─────────────────────────────────────────────
@st.dialog("🔥 Sering Dibeli Bersamaan", width="large")
def show_recommendation_popup(added_item: str, rec_rows: list):
    st.markdown(f"""
    <div style="margin-bottom:1rem;">
        <span style="font-size:0.85rem; color:#6b7699;">
            Pelanggan yang membeli
            <b style='color:#a78bfa'>{katalog[added_item]['emoji']} {added_item}</b>
            juga sering membeli:
        </span>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(min(len(rec_rows), 4))
    for i, (iname, row) in enumerate(rec_rows[:4]):
        with cols[i]:
            conf_pct = row["confidence"] * 100
            st.markdown(f"""
            <div style="background:#0d1120; border:1px solid #1e2540; border-radius:12px;
                        overflow:hidden; text-align:center;">
                <img src="{katalog[iname]['img']}"
                     style="width:100%; height:100px; object-fit:cover;" />
                <div style="padding:0.7rem;">
                    <div style="font-size:1.3rem;">{katalog[iname]['emoji']}</div>
                    <div style="font-weight:700; font-size:0.88rem; color:#e8eaf0; margin:0.15rem 0;">
                        {iname}
                    </div>
                    <div style="font-family:'JetBrains Mono'; font-size:0.75rem; color:#34d399;">
                        {katalog[iname]['harga_str']}
                    </div>
                    <div style="font-size:0.7rem; color:#6b7699; margin-top:0.3rem;">
                        Kecocokan
                        <span style="color:#a78bfa; font-weight:700;">{conf_pct:.0f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"+ Tambah {iname}", key=f"popup_add_{iname}"):
                st.session_state.keranjang[iname] = (
                    st.session_state.keranjang.get(iname, 0) + 1
                )
                st.session_state.show_popup = False
                st.rerun()

    st.markdown("<div style='margin-top:1rem'>", unsafe_allow_html=True)
    if st.button("Tutup", key="popup_close"):
        st.session_state.show_popup = False
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding:2rem 0 1rem; text-align:center;">
    <div style="font-size:0.72rem; font-weight:700; letter-spacing:0.2em; color:#a78bfa;
                text-transform:uppercase; margin-bottom:0.5rem;">
        Laboratorium Data Mining
    </div>
    <h1 style="font-family:'Sora',sans-serif; font-size:2.4rem; font-weight:800;
               background:linear-gradient(135deg,#e8eaf0 30%,#a78bfa);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               margin:0; letter-spacing:-0.02em;">
        🛒 DataMart
    </h1>
    <p style="color:#6b7699; font-size:0.88rem; margin-top:0.4rem;">
        Toko Cerdas Berbasis
        <span style="color:#a78bfa; font-weight:600;">Association Rule Mining</span>
    </p>
</div>
""", unsafe_allow_html=True)

tab_store, tab_admin = st.tabs(["🛍️  Toko", "🔒  Admin"])

# ══════════════════════════════════════════════════════
# TAB 1 — TOKO
# ══════════════════════════════════════════════════════
with tab_store:

    # ── Popup rekomendasi ────────────────────────────────
    if st.session_state.show_popup and st.session_state.popup_item:
        df_s = load_sales()
        _, rules_live = run_apriori(df_s, min_sup=0.2, min_conf=0.5, min_lift=1.0)
        cart_items = list(st.session_state.keranjang.keys())
        rec_rows = get_recommendations(cart_items, rules_live)
        # rec_rows selalu list — aman dievaluasi sebagai boolean
        if rec_rows:
            show_recommendation_popup(st.session_state.popup_item, rec_rows)
        else:
            st.session_state.show_popup = False

    # ── Katalog produk ───────────────────────────────────
    st.markdown("""
    <div style="margin:1.5rem 0 0.75rem;">
        <h3 style="font-size:1.05rem; font-weight:700; color:#e8eaf0; margin:0;">
            Produk Pilihan Hari Ini
        </h3>
        <p style="color:#6b7699; font-size:0.78rem; margin-top:0.2rem;">
            Klik <b style='color:#a78bfa'>+ Keranjang</b> untuk menambahkan produk
        </p>
    </div>
    """, unsafe_allow_html=True)

    prod_cols = st.columns(5)
    for idx, (nama, info) in enumerate(katalog.items()):
        with prod_cols[idx]:
            in_cart = nama in st.session_state.keranjang
            border = "#a78bfa" if in_cart else "#1e2540"
            bg     = "#1a1040" if in_cart else "#111827"
            badge  = (
                f'<div style="font-size:0.68rem; color:#a78bfa; margin-top:0.2rem;">'
                f'✓ ×{st.session_state.keranjang[nama]}</div>'
            ) if in_cart else ""

            st.markdown(f"""
            <div style="background:{bg}; border:1px solid {border}; border-radius:14px;
                        overflow:hidden; margin-bottom:0.5rem;">
                <img src="{info['img']}"
                     style="width:100%; height:125px; object-fit:cover;" />
                <div style="padding:0.6rem;">
                    <div style="font-size:0.68rem; color:#a78bfa; font-weight:600;
                                margin-bottom:0.12rem;">
                        {info['emoji']} {info['desc']}
                    </div>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:0.8rem;
                                font-weight:700; color:#34d399;">
                        {info['harga_str']}
                    </div>
                    {badge}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("+ Keranjang", key=f"add_{nama}"):
                st.session_state.keranjang[nama] = (
                    st.session_state.keranjang.get(nama, 0) + 1
                )
                st.session_state.checkout_done = False
                st.session_state.popup_item    = nama
                st.session_state.show_popup    = True
                st.rerun()

    st.divider()

    # ── Keranjang & info ─────────────────────────────────
    col_cart, col_info = st.columns([1, 2], gap="large")

    with col_cart:
        st.markdown(
            '<h3 style="font-size:1.02rem; font-weight:700; color:#e8eaf0; '
            'margin-bottom:0.75rem;">🛒 Keranjang Belanja</h3>',
            unsafe_allow_html=True,
        )

        if not st.session_state.keranjang:
            st.markdown("""
            <div style="background:#111827; border:1px dashed #1e2540; border-radius:12px;
                        padding:2rem; text-align:center; color:#6b7699;">
                <div style="font-size:2rem; margin-bottom:0.4rem;">🛒</div>
                <div style="font-size:0.82rem;">Keranjang masih kosong</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            total = 0
            for item in list(st.session_state.keranjang.keys()):
                qty      = st.session_state.keranjang[item]
                harga    = katalog[item]["harga"]
                subtotal = harga * qty
                total   += subtotal

                ci, cm, cq, cx = st.columns([3, 1, 1, 1])
                with ci:
                    st.markdown(f"""
                    <div style="padding:0.5rem; background:#111827; border:1px solid #1e2540;
                                border-radius:8px; margin-bottom:0.3rem;">
                        <div style="font-size:0.83rem; font-weight:600; color:#e8eaf0;">
                            {katalog[item]['emoji']} {item}
                        </div>
                        <div style="font-size:0.7rem; font-family:'JetBrains Mono'; color:#34d399;">
                            {qty} × Rp {harga:,} = <b>Rp {subtotal:,}</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with cm:
                    st.markdown('<div style="margin-top:0.1rem">', unsafe_allow_html=True)
                    if st.button("−", key=f"dec_{item}"):
                        if st.session_state.keranjang[item] > 1:
                            st.session_state.keranjang[item] -= 1
                        else:
                            del st.session_state.keranjang[item]
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                with cq:
                    st.markdown('<div style="margin-top:0.1rem">', unsafe_allow_html=True)
                    if st.button("+", key=f"inc_{item}"):
                        st.session_state.keranjang[item] += 1
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                with cx:
                    st.markdown('<div style="margin-top:0.1rem">', unsafe_allow_html=True)
                    if st.button("✕", key=f"rm_{item}"):
                        del st.session_state.keranjang[item]
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:#1a1040; border:1px solid #a78bfa; border-radius:10px;
                        padding:0.7rem 1rem; margin:0.6rem 0;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:#a78bfa; font-weight:600; font-size:0.83rem;">
                        Total ({sum(st.session_state.keranjang.values())} item)
                    </span>
                    <span style="font-family:'JetBrains Mono'; font-size:1.05rem;
                                 font-weight:700; color:#e8eaf0;">
                        Rp {total:,}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            co_col, cl_col = st.columns(2)
            with co_col:
                if st.button("✅ Checkout", key="checkout_btn"):
                    tid = save_transaction(st.session_state.keranjang)
                    st.session_state.last_tid      = tid
                    st.session_state.checkout_done = True
                    st.session_state.keranjang     = {}
                    st.session_state.show_popup    = False
                    st.rerun()
            with cl_col:
                if st.button("🗑️ Kosongkan", key="clear_btn"):
                    st.session_state.keranjang     = {}
                    st.session_state.checkout_done = False
                    st.session_state.show_popup    = False
                    st.rerun()

        if st.session_state.checkout_done and st.session_state.last_tid:
            st.success(
                f"✅ Transaksi #{st.session_state.last_tid} berhasil! "
                f"Data tersimpan ke `penjualan.csv`"
            )

    with col_info:
        st.markdown("""
        <div style="background:#111827; border:1px solid #1e2540; border-radius:14px;
                    padding:1.5rem; margin-top:0.5rem;">
            <div style="font-size:0.75rem; font-weight:700; color:#a78bfa;
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:1rem;">
                💡 Cara Belanja
            </div>
        """, unsafe_allow_html=True)

        tips = [
            ("🛍️", "Klik <b>+ Keranjang</b> untuk menambah produk ke keranjang."),
            ("🔥", "Sistem akan otomatis menampilkan <b>popup rekomendasi</b> produk "
                   "terkait berdasarkan histori pembelian."),
            ("➕➖", "Gunakan tombol <b>+</b> dan <b>−</b> di keranjang untuk mengubah kuantitas."),
            ("✅", "Klik <b>Checkout</b> untuk menyelesaikan transaksi. "
                   "Data kuantitas akan tersimpan otomatis."),
        ]
        for icon, tip in tips:
            st.markdown(f"""
            <div style="display:flex; gap:0.75rem; padding:0.6rem 0;
                        border-bottom:1px solid #1e2540; align-items:flex-start;">
                <span style="font-size:1.2rem; flex-shrink:0;">{icon}</span>
                <span style="font-size:0.82rem; color:#c4c8d8; line-height:1.5;">{tip}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 2 — ADMIN
# ══════════════════════════════════════════════════════
with tab_admin:

    if not st.session_state.admin_logged:
        st.markdown("""
        <div style="max-width:380px; margin:3rem auto; text-align:center;">
            <div style="font-size:3rem; margin-bottom:0.75rem;">🔒</div>
            <h2 style="font-size:1.35rem; font-weight:700; color:#e8eaf0; margin-bottom:0.4rem;">
                Akses Admin
            </h2>
            <p style="color:#6b7699; font-size:0.82rem; margin-bottom:1.5rem;">
                Masukkan kredensial untuk masuk ke dashboard
            </p>
        </div>
        """, unsafe_allow_html=True)

        lc = st.columns([1, 2, 1])[1]
        with lc:
            uname  = st.text_input("Username", placeholder="datmin",   key="au")
            passwd = st.text_input("Password", type="password",
                                   placeholder="••••••••", key="ap")
            if st.button("Masuk ke Dashboard Admin", key="login_btn"):
                if uname == ADMIN_USER and passwd == ADMIN_PASS:
                    st.session_state.admin_logged = True
                    st.rerun()
                else:
                    st.error("❌ Username atau password salah.")

    else:
        # ── Header ──────────────────────────────────────────
        hc, lc = st.columns([6, 1])
        with hc:
            st.markdown("""
            <h2 style="font-size:1.35rem; font-weight:800; color:#e8eaf0; margin:1rem 0 0.2rem;">
                📊 Dashboard Admin — DataMart
            </h2>
            <p style="color:#6b7699; font-size:0.78rem; margin-bottom:1.2rem;">
                Analisis penjualan &amp; Association Rule Mining
            </p>
            """, unsafe_allow_html=True)
        with lc:
            st.markdown("<div style='margin-top:1.5rem'>", unsafe_allow_html=True)
            if st.button("🚪 Logout", key="logout_btn"):
                st.session_state.admin_logged = False
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        df_admin = load_sales()

        if df_admin.empty or len(df_admin) < 1:
            st.warning(
                "⚠️ Belum ada transaksi. "
                "Lakukan checkout di halaman Toko terlebih dahulu."
            )
        else:
            item_cols = [c for c in ITEM_NAMES if c in df_admin.columns]

            # ── METRICS ───────────────────────────────────────────
            st.markdown("### 📈 Ringkasan Statistik")
            m1, m2, m3, m4, m5 = st.columns(5)
            total_trx  = len(df_admin)
            total_units = int(df_admin[item_cols].sum().sum())
            avg_basket  = df_admin[item_cols].gt(0).sum(axis=1).mean()
            top_item    = df_admin[item_cols].sum().idxmax()
            revenue     = sum(
                df_admin[c].sum() * katalog[c]["harga"] for c in item_cols
            )

            with m1: st.metric("Total Transaksi",    f"{total_trx:,}")
            with m2: st.metric("Total Unit Terjual", f"{total_units:,}")
            with m3: st.metric("Avg Jenis / Transaksi", f"{avg_basket:.1f}")
            with m4: st.metric("Produk Terlaris",
                               f"{katalog[top_item]['emoji']} {top_item}")
            with m5: st.metric("Est. Revenue", f"Rp {revenue:,.0f}")

            st.divider()

            # ── VISUALISASI ───────────────────────────────────────
            st.markdown("### 📊 Visualisasi Bisnis")
            vc1, vc2 = st.columns(2)

            with vc1:
                units_per = df_admin[item_cols].sum().sort_values(ascending=False)
                fig1 = go.Figure(go.Bar(
                    x=units_per.index.tolist(), y=units_per.values.tolist(),
                    marker=dict(color=COLORS, line=dict(width=0)),
                    text=units_per.values.tolist(), textposition="outside",
                    textfont=dict(color="#e8eaf0", size=11)
                ))
                fig1.update_layout(
                    title=dict(text="Total Unit Terjual per Produk",
                               font=dict(color="#e8eaf0", size=13)),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#6b7699", family="Sora"),
                    xaxis=dict(gridcolor="#1e2540"), yaxis=dict(gridcolor="#1e2540"),
                    margin=dict(t=45, b=20, l=10, r=10), height=300
                )
                st.plotly_chart(fig1, use_container_width=True)

            with vc2:
                revs = {
                    item: int(df_admin[item].sum() * katalog[item]["harga"])
                    for item in item_cols
                }
                fig2 = go.Figure(go.Pie(
                    labels=list(revs.keys()), values=list(revs.values()), hole=0.5,
                    marker=dict(colors=COLORS, line=dict(color="#0a0e1a", width=2)),
                    textfont=dict(color="#e8eaf0", size=11)
                ))
                fig2.update_layout(
                    title=dict(text="Distribusi Estimasi Revenue",
                               font=dict(color="#e8eaf0", size=13)),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#6b7699", family="Sora"),
                    legend=dict(font=dict(color="#e8eaf0")),
                    margin=dict(t=45, b=10, l=10, r=10), height=300
                )
                st.plotly_chart(fig2, use_container_width=True)

            vc3, vc4 = st.columns(2)

            with vc3:
                basket_sizes = df_admin[item_cols].gt(0).sum(axis=1)
                fig3 = px.histogram(
                    basket_sizes, nbins=6,
                    color_discrete_sequence=["#a78bfa"],
                    labels={"value": "Jenis Item", "count": "Frekuensi"}
                )
                fig3.update_layout(
                    title=dict(text="Distribusi Basket Size (Jenis Item)",
                               font=dict(color="#e8eaf0", size=13)),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#6b7699", family="Sora"),
                    xaxis=dict(gridcolor="#1e2540"), yaxis=dict(gridcolor="#1e2540"),
                    showlegend=False,
                    margin=dict(t=45, b=20, l=10, r=10), height=300
                )
                st.plotly_chart(fig3, use_container_width=True)

            with vc4:
                bool_df = df_admin[item_cols].gt(0).astype(int)
                cooc    = bool_df.T.dot(bool_df)
                cooc_arr = cooc.values.copy()
                np.fill_diagonal(cooc_arr, 0)
                cooc = pd.DataFrame(cooc_arr, index=cooc.index, columns=cooc.columns)
                fig4 = go.Figure(go.Heatmap(
                    z=cooc.values,
                    x=cooc.columns.tolist(), y=cooc.index.tolist(),
                    colorscale=[[0,"#0a0e1a"],[0.5,"#7c3aed"],[1,"#a78bfa"]],
                    text=cooc.values, texttemplate="%{text}",
                    textfont=dict(color="white", size=12), showscale=False
                ))
                fig4.update_layout(
                    title=dict(text="Co-occurrence Produk",
                               font=dict(color="#e8eaf0", size=13)),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#6b7699", family="Sora"),
                    margin=dict(t=45, b=20, l=10, r=10), height=300
                )
                st.plotly_chart(fig4, use_container_width=True)

            st.divider()

            # ── STATISTIK DESKRIPTIF ──────────────────────────────
            st.markdown("### 🔢 Statistik Deskriptif")
            sd1, sd2 = st.columns([1, 2])

            with sd1:
                desc = (
                    df_admin[item_cols]
                    .describe().T[["count","mean","std","min","max"]]
                    .round(3)
                )
                desc.index.name = "Produk"
                desc.columns    = ["N Trx", "Mean Qty", "Std", "Min", "Max"]
                st.dataframe(desc, use_container_width=True)

            with sd2:
                st.markdown("""
                <div style="background:#111827; border:1px solid #1e2540;
                            border-radius:12px; padding:1.2rem;">
                    <div style="font-size:0.75rem; font-weight:700; color:#a78bfa;
                                margin-bottom:0.75rem; text-transform:uppercase;
                                letter-spacing:0.1em;">
                        💡 Insight Bisnis Otomatis
                    </div>
                """, unsafe_allow_html=True)

                ifreq  = df_admin[item_cols].gt(0).sum().sort_values(ascending=False)
                iunits = df_admin[item_cols].sum().sort_values(ascending=False)
                insights = [
                    f"🏆 <b>{ifreq.index[0]}</b> paling sering dibeli — muncul di "
                    f"<b>{int(ifreq.iloc[0])}</b> transaksi "
                    f"({ifreq.iloc[0]/total_trx*100:.0f}%).",
                    f"📦 <b>{iunits.index[0]}</b> memiliki total unit terjual tertinggi: "
                    f"<b>{int(iunits.iloc[0])}</b> unit.",
                    f"📉 <b>{ifreq.index[-1]}</b> paling jarang dibeli "
                    f"({int(ifreq.iloc[-1])} transaksi). Pertimbangkan promosi bundling.",
                ]
                if avg_basket >= 2:
                    insights.append(
                        f"🛒 Rata-rata pelanggan membeli <b>{avg_basket:.1f} jenis item</b>"
                        f" — cross-selling berjalan baik."
                    )
                else:
                    insights.append(
                        f"🛒 Rata-rata basket size hanya <b>{avg_basket:.1f} jenis</b>. "
                        f"Optimalkan rekomendasi produk."
                    )

                cooc_c = cooc.copy()
                cooc_c_arr = cooc_c.values.copy()
                np.fill_diagonal(cooc_c_arr, 0)
                cooc_c = pd.DataFrame(cooc_c_arr, index=cooc_c.index, columns=cooc_c.columns)
                if cooc_c.values.max() > 0:
                    mi = np.unravel_index(cooc_c.values.argmax(), cooc_c.shape)
                    a_it = cooc_c.index[mi[0]]
                    b_it = cooc_c.columns[mi[1]]
                    insights.append(
                        f"🔗 Pasangan paling sering dibeli bersama: "
                        f"<b>{a_it} + {b_it}</b> ({int(cooc_c.values[mi])} transaksi)."
                    )

                for ins in insights:
                    st.markdown(
                        f'<div style="padding:0.45rem 0; border-bottom:1px solid #1e2540; '
                        f'font-size:0.8rem; color:#c4c8d8; line-height:1.5;">{ins}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            st.divider()

            # ── APRIORI ───────────────────────────────────────────
            st.markdown("### 🔬 Analisis Association Rules (Apriori)")
            ap1, ap2, ap3 = st.columns(3)
            with ap1:
                a_sup = st.slider(
                    "Min Support", 0.05, 0.9, 0.2, 0.05,
                    key="adm_sup",
                    help="Frekuensi minimum itemset muncul di seluruh transaksi"
                )
            with ap2:
                a_conf = st.slider(
                    "Min Confidence", 0.1, 1.0, 0.5, 0.05,
                    key="adm_conf",
                    help="Tingkat kepercayaan: P(B|A)"
                )
            with ap3:
                a_lift = st.slider(
                    "Min Lift", 1.0, 5.0, 1.0, 0.1,
                    key="adm_lift",
                    help="Lift > 1 artinya hubungan positif antar item"
                )

            freq_a, rules_a = run_apriori(df_admin, a_sup, a_conf, a_lift)
            if rules_a is not None and not rules_a.empty:
                st.success(
                    f"✅ Ditemukan **{len(rules_a)}** aturan asosiasi "
                    f"dari **{len(freq_a)}** frequent itemsets."
                )
                dr = rules_a[[
                    "antecedents","consequents","support",
                    "confidence","lift","leverage","conviction"
                ]].copy()
                dr["antecedents"] = dr["antecedents"].apply(
                    lambda x: " + ".join(list(x))
                )
                dr["consequents"] = dr["consequents"].apply(
                    lambda x: " + ".join(list(x))
                )
                dr["support"]    = (dr["support"]    * 100).round(2).astype(str) + "%"
                dr["confidence"] = (dr["confidence"] * 100).round(2).astype(str) + "%"
                dr["lift"]       = dr["lift"].round(4)
                dr["leverage"]   = dr["leverage"].round(4)
                dr["conviction"] = dr["conviction"].round(4)
                dr.columns = [
                    "Antecedents","Consequents","Support",
                    "Confidence","Lift","Leverage","Conviction"
                ]
                st.dataframe(dr, use_container_width=True, hide_index=True)
            else:
                n = len(df_admin)
                if n < 3:
                    st.warning(
                        f"Hanya ada **{n} transaksi**. "
                        f"Butuh minimal 3 untuk menghasilkan aturan asosiasi."
                    )
                else:
                    st.info(
                        "Tidak ada aturan dengan parameter saat ini. "
                        "Coba turunkan Min Support atau Min Confidence."
                    )

            st.divider()

            # ── DATASET MENTAH ────────────────────────────────────
            st.markdown("### 📂 Dataset Mentah History Penjualan")
            dc, di = st.columns([1, 3])
            with di:
                st.caption(
                    f"📄 Google Sheets | {len(df_admin)} transaksi "
                    f"| Format: kuantitas nyata per produk"
                )
            with dc:
                st.download_button(
                    "⬇️ Download CSV",
                    df_admin.to_csv(index=False).encode("utf-8"),
                    "penjualan.csv", "text/csv", key="dl_csv"
                )
            st.dataframe(
                df_admin.style.background_gradient(subset=item_cols, cmap="Purples"),
                use_container_width=True, hide_index=True
            )