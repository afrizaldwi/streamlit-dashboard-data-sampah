"""
app.py — Dashboard Pengelolaan Sampah Indonesia

Dashboard requirement:
- Streamlit UI
- Pandas data processing
- Plotly interactive charts
- Optional Plotly choropleth map using Indonesia province-level GeoJSON

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Dashboard Pengelolaan Sampah Indonesia",
    page_icon="bar-chart",
    layout="wide",
)


# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

DATA_PATHS = [
    Path("data/processed/sampah_clean.csv"),
    Path("data/sampah_clean.csv"),
    Path("sampah_clean.csv"),
]

GEOJSON_PATHS = [
    Path("data/geo/indonesia_provinces.geojson"),
    Path("data/indonesia_provinces.geojson"),
    Path("indonesia_provinces.geojson"),
]

REQUIRED_COLUMNS = [
    "tahun",
    "provinsi",
    "kabupaten_kota",
    "timbulan_tahunan_ton",
    "terkelola_ton",
    "gap_tidak_tertangani_ton",
    "pct_terkelola",
    "kategori_pengelolaan",
]

NUMERIC_COLUMNS = [
    "timbulan_harian_ton",
    "timbulan_tahunan_ton",
    "pengurangan_ton",
    "pct_pengurangan",
    "penanganan_ton",
    "pct_penanganan",
    "terkelola_ton",
    "pct_terkelola",
    "daur_ulang_ton",
    "bahan_baku_ton",
    "recycling_rate",
    "komposisi_makanan_pct",
    "komposisi_kayu_pct",
    "komposisi_kertas_pct",
    "komposisi_plastik_pct",
    "komposisi_logam_pct",
    "komposisi_kain_pct",
    "komposisi_karet_pct",
    "komposisi_kaca_pct",
    "komposisi_lainnya_pct",
    "gap_tidak_tertangani_ton",
]

COMPOSITION_COLUMNS = {
    "komposisi_makanan_pct": "Sisa makanan",
    "komposisi_plastik_pct": "Plastik",
    "komposisi_kertas_pct": "Kertas/Karton",
    "komposisi_kayu_pct": "Kayu/Ranting",
    "komposisi_logam_pct": "Logam",
    "komposisi_kain_pct": "Kain",
    "komposisi_karet_pct": "Karet/Kulit",
    "komposisi_kaca_pct": "Kaca",
    "komposisi_lainnya_pct": "Lainnya",
}

CATEGORY_ORDER = [
    "Sangat Kurang (<30%)",
    "Kurang (30-60%)",
    "Cukup (60-80%)",
    "Baik (>80%)",
]

CATEGORY_COLORS = {
    "Sangat Kurang (<30%)": "#f43f5e",
    "Kurang (30-60%)": "#fb923c",
    "Cukup (60-80%)": "#38bdf8",
    "Baik (>80%)": "#22c55e",
}

MAP_METRICS = {
    "Timbulan sampah tahunan": "timbulan_tahunan_ton",
    "Sampah terkelola": "terkelola_ton",
    "Gap tidak tertangani": "gap_tidak_tertangani_ton",
    "Persentase pengelolaan": "management_rate",
}

ALL_INDONESIA_LABEL = "Seluruh Indonesia"

ALL_KABUPATEN_LABEL = "Semua Kabupaten/Kota"

MAP_COLOR_SCALE = ["#ecfeff", "#a5f3fc", "#22d3ee", "#0891b2", "#164e63"]
REGIONAL_COLOR_SCALE = ["#f0fdfa", "#99f6e4", "#2dd4bf", "#0f766e", "#134e4a"]
COMPOSITION_COLORS = [
    "#14b8a6",
    "#0ea5e9",
    "#6366f1",
    "#a855f7",
    "#f97316",
    "#84cc16",
    "#eab308",
    "#64748b",
    "#ec4899",
]
TREND_COLORS = {
    "Timbulan sampah": "#38bdf8",
    "Sampah terkelola": "#22c55e",
}


# -------------------------------------------------------------------
# Theme-aware styling
# -------------------------------------------------------------------

theme_base = st.get_option("theme.base") or "light"
is_dark_mode = theme_base.lower() == "dark"

TEXT_COLOR = "#f8fafc" if is_dark_mode else "#101828"
MUTED_TEXT_COLOR = "#cbd5e1" if is_dark_mode else "#667085"
NOTE_BG_COLOR = "#111827" if is_dark_mode else "#f8fafc"
NOTE_BORDER_COLOR = "#334155" if is_dark_mode else "#e5e7eb"
NOTE_TEXT_COLOR = "#dbeafe" if is_dark_mode else "#344054"
METRIC_BG_COLOR = "#111827" if is_dark_mode else "#ffffff"
METRIC_BORDER_COLOR = "#334155" if is_dark_mode else "#e5e7eb"
METRIC_LABEL_COLOR = "#cbd5e1" if is_dark_mode else "#667085"
METRIC_VALUE_COLOR = "#f8fafc" if is_dark_mode else "#101828"
GRID_COLOR = (
    "rgba(148, 163, 184, 0.22)" if is_dark_mode else "rgba(148, 163, 184, 0.25)"
)
PLOTLY_TEMPLATE = "plotly_dark" if is_dark_mode else "plotly_white"

st.markdown(
    f"""
    <style>
    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }}
    .dashboard-title {{
        color: {TEXT_COLOR};
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.25rem;
    }}
    .dashboard-subtitle {{
        color: {MUTED_TEXT_COLOR};
        font-size: 0.98rem;
        margin-bottom: 1.2rem;
    }}
    .section-title {{
        color: {TEXT_COLOR};
        font-size: 1.25rem;
        font-weight: 750;
        margin-top: 1.25rem;
        margin-bottom: 0.4rem;
    }}
    .note-box {{
        background: {NOTE_BG_COLOR};
        border: 1px solid {NOTE_BORDER_COLOR};
        border-radius: 0.75rem;
        padding: 0.9rem 1rem;
        color: {NOTE_TEXT_COLOR};
        font-size: 0.93rem;
    }}
    div[data-testid="stMetric"] {{
        background: {METRIC_BG_COLOR};
        border: 1px solid {METRIC_BORDER_COLOR};
        border-radius: 0.85rem;
        padding: 1rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.10);
        color: {METRIC_VALUE_COLOR} !important;
    }}
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] p {{
        color: {METRIC_LABEL_COLOR} !important;
    }}
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] div {{
        color: {METRIC_VALUE_COLOR} !important;
    }}
    div[data-testid="stMetric"] svg {{
        color: {METRIC_LABEL_COLOR} !important;
        fill: {METRIC_LABEL_COLOR} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------


def safe_sum(series: pd.Series) -> float | np.nan:
    if series.dropna().empty:
        return np.nan
    return series.sum(min_count=1)


def calculate_management_rate(
    generated: float | np.nan, managed: float | np.nan
) -> float | np.nan:
    if pd.notna(generated) and generated > 0 and pd.notna(managed):
        return (managed / generated) * 100
    return np.nan


def interpret_correlation(value: float) -> str:
    if pd.isna(value):
        return "Tidak dapat dihitung"
    abs_val = abs(value)
    if abs_val < 0.2:
        return "sangat lemah"
    elif abs_val < 0.4:
        return "lemah"
    elif abs_val < 0.6:
        return "sedang"
    elif abs_val < 0.8:
        return "kuat"
    else:
        return "sangat kuat"


def select_ranked_rows(
    data: pd.DataFrame, metric: str, mode: str, count: int
) -> pd.DataFrame:
    if data.empty or metric not in data.columns:
        return data
    valid_data = data.dropna(subset=[metric])
    if valid_data.empty:
        return valid_data
    ascending = mode == "Terendah"
    return valid_data.sort_values(metric, ascending=ascending).head(count)


def first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_data_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_data() -> pd.DataFrame:
    path = first_existing_path(DATA_PATHS)
    if path is None:
        st.error(
            "File CSV tidak ditemukan. Letakkan `sampah_clean.csv` di salah satu lokasi berikut: "
            "`data/processed/sampah_clean.csv`, `data/sampah_clean.csv`, atau `sampah_clean.csv`."
        )
        st.stop()
    return load_data_from_path(str(path))


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    missing_required = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing_required:
        st.error(f"Kolom wajib tidak ditemukan: {', '.join(missing_required)}")
        st.stop()

    data["tahun"] = pd.to_numeric(data["tahun"], errors="coerce").astype("Int64")
    for col in NUMERIC_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data["provinsi"] = data["provinsi"].astype(str).str.strip()
    data["kabupaten_kota"] = data["kabupaten_kota"].astype(str).str.strip()

    data = data.dropna(subset=["tahun", "provinsi"])
    data["tahun"] = data["tahun"].astype(int)

    return data


def normalize_province_name(name: Any) -> str:
    if name is None or pd.isna(name):
        return ""
    text = str(name).upper().strip()
    text = re.sub(r"\b(PROVINSI|PROPINSI|PROVINCE|PROV\.?)\b", " ", text)
    text = text.replace("&", "DAN")
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    aliases = {
        "NANGGROE ACEH DARUSSALAM": "ACEH",
        "DAERAH ISTIMEWA ACEH": "ACEH",
        "DI ACEH": "ACEH",
        "SUMATERA UTARA": "SUMATERA UTARA",
        "SUMATRA UTARA": "SUMATERA UTARA",
        "SUMATERA BARAT": "SUMATERA BARAT",
        "SUMATRA BARAT": "SUMATERA BARAT",
        "SUMATERA SELATAN": "SUMATERA SELATAN",
        "SUMATRA SELATAN": "SUMATERA SELATAN",
        "DAERAH KHUSUS IBUKOTA JAKARTA": "DKI JAKARTA",
        "DKI JAKARTA RAYA": "DKI JAKARTA",
        "JAKARTA RAYA": "DKI JAKARTA",
        "JAKARTA": "DKI JAKARTA",
        "D I YOGYAKARTA": "DI YOGYAKARTA",
        "DI YOGYAKARTA": "DI YOGYAKARTA",
        "DAERAH ISTIMEWA YOGYAKARTA": "DI YOGYAKARTA",
        "YOGYAKARTA": "DI YOGYAKARTA",
        "KEP RIAU": "KEPULAUAN RIAU",
        "KEPULAUAN RIAU": "KEPULAUAN RIAU",
        "RIAU KEPULAUAN": "KEPULAUAN RIAU",
        "BANGKA BELITUNG": "KEPULAUAN BANGKA BELITUNG",
        "KEP BANGKA BELITUNG": "KEPULAUAN BANGKA BELITUNG",
        "KEPULAUAN BANGKA BELITUNG": "KEPULAUAN BANGKA BELITUNG",
        "NTB": "NUSA TENGGARA BARAT",
        "NUSA TENGGARA BARAT": "NUSA TENGGARA BARAT",
        "NUSA TENGGARA TIMUR": "NUSA TENGGARA TIMUR",
        "NTT": "NUSA TENGGARA TIMUR",
        "KALIMANTAN UTARA": "KALIMANTAN UTARA",
        "KALTARA": "KALIMANTAN UTARA",
        "PAPUA BARAT DAYA": "PAPUA BARAT DAYA",
        "SOUTHWEST PAPUA": "PAPUA BARAT DAYA",
        "PAPUA PEGUNUNGAN": "PAPUA PEGUNUNGAN",
        "HIGHLAND PAPUA": "PAPUA PEGUNUNGAN",
        "PAPUA TENGAH": "PAPUA TENGAH",
        "CENTRAL PAPUA": "PAPUA TENGAH",
        "PAPUA SELATAN": "PAPUA SELATAN",
        "SOUTH PAPUA": "PAPUA SELATAN",
    }
    return aliases.get(text, text)


def format_decimal_id(value: float | int, decimals: int = 2) -> str:
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def format_tons(value: float | int | None) -> str:
    if pd.isna(value):
        return "Tidak tersedia"
    abs_value = abs(float(value))
    if abs_value >= 1_000_000:
        return f"{format_decimal_id(value / 1_000_000, 2)} juta ton"
    if abs_value >= 1_000:
        return f"{format_decimal_id(value / 1_000, 2)} ribu ton"
    return f"{format_decimal_id(value, 0)} ton"


def format_number(value: float | int | None) -> str:
    if pd.isna(value):
        return "Tidak tersedia"
    return format_decimal_id(value, 0)


def format_percent(value: float | int | None) -> str:
    if pd.isna(value):
        return "Tidak tersedia"
    return f"{format_decimal_id(value, 2)}%"


def aggregate_province(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby("provinsi", as_index=False).agg(
        timbulan_tahunan_ton=("timbulan_tahunan_ton", lambda x: x.sum(min_count=1)),
        terkelola_ton=("terkelola_ton", lambda x: x.sum(min_count=1)),
        gap_tidak_tertangani_ton=(
            "gap_tidak_tertangani_ton",
            lambda x: x.sum(min_count=1),
        ),
        pengurangan_ton=("pengurangan_ton", lambda x: x.sum(min_count=1)),
        penanganan_ton=("penanganan_ton", lambda x: x.sum(min_count=1)),
        daur_ulang_ton=("daur_ulang_ton", lambda x: x.sum(min_count=1)),
        pct_terkelola=("pct_terkelola", "mean"),
        recycling_rate=("recycling_rate", "mean"),
        total_kabupaten_kota=("kabupaten_kota", "nunique"),
    )

    grouped["management_rate"] = np.where(
        (grouped["timbulan_tahunan_ton"] > 0) & (grouped["terkelola_ton"].notna()),
        grouped["terkelola_ton"] / grouped["timbulan_tahunan_ton"] * 100,
        np.nan,
    )

    grouped["prov_norm"] = grouped["provinsi"].apply(normalize_province_name)
    return grouped


def make_empty_figure(message: str, height: int = 420) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=15, color=TEXT_COLOR),
        align="center",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def apply_chart_theme(fig: go.Figure, height: int | None = None) -> go.Figure:
    layout_update: dict[str, Any] = {
        "template": PLOTLY_TEMPLATE,
        "font": dict(color=TEXT_COLOR),
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "xaxis": dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        "yaxis": dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    }
    if height is not None:
        layout_update["height"] = height

    fig.update_layout(**layout_update)
    return fig


@st.cache_data(show_spinner=False)
def load_geojson_from_path(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_geojson() -> dict[str, Any] | None:
    path = first_existing_path(GEOJSON_PATHS)
    if path is None:
        return None
    try:
        return load_geojson_from_path(str(path))
    except Exception as exc:
        st.sidebar.error(f"File GeoJSON tidak bisa dibaca: {exc}")
        return None


def detect_geojson_name_property(
    geojson: dict[str, Any],
    csv_norm_names: set[str],
) -> str | None:
    features = geojson.get("features", [])
    if not features:
        return None

    candidate_keys = [
        "provinsi",
        "Provinsi",
        "PROVINSI",
        "PROPINSI",
        "Propinsi",
        "province",
        "Province",
        "NAME_1",
        "NAME",
        "name",
        "shapeName",
        "WADMPR",
        "ADM1_EN",
        "ADM1_ID",
    ]

    first_properties = features[0].get("properties", {})
    for key in candidate_keys:
        if key in first_properties:
            return key

    best_key = None
    best_matches = 0

    for key in first_properties.keys():
        matches = 0
        for feature in features:
            value = feature.get("properties", {}).get(key)
            if (
                isinstance(value, str)
                and normalize_province_name(value) in csv_norm_names
            ):
                matches += 1
        if matches > best_matches:
            best_key = key
            best_matches = matches

    return best_key if best_matches > 0 else None


def prepare_geojson(
    geojson: dict[str, Any],
    csv_norm_names: set[str],
) -> tuple[dict[str, Any], str | None, list[str]]:
    property_name = detect_geojson_name_property(geojson, csv_norm_names)
    if property_name is None:
        return geojson, None, sorted(csv_norm_names)

    features = geojson.get("features", [])
    geo_norm_names = set()

    for feature in features:
        properties = feature.setdefault("properties", {})
        normalized = normalize_province_name(properties.get(property_name))
        properties["prov_norm"] = normalized
        feature["id"] = normalized
        geo_norm_names.add(normalized)

    unmatched_csv_names = sorted(csv_norm_names - geo_norm_names)
    return geojson, property_name, unmatched_csv_names


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------

with st.sidebar:
    st.header("Filter")

raw_df = load_data()
df = clean_data(raw_df)

years = sorted(df["tahun"].dropna().astype(int).unique(), reverse=True)
provinces = sorted(df["provinsi"].dropna().unique().tolist())

latest_year = years[0] if years else 2024


# -------------------------------------------------------------------
# Sidebar filters
# -------------------------------------------------------------------

with st.sidebar:
    selected_year = st.selectbox(
        "Tahun",
        options=years,
        index=years.index(latest_year),
    )

    if "prov_selections" not in st.session_state:
        st.session_state.prov_selections = {p: True for p in provinces}

    def update_prov(p):
        st.session_state.prov_selections[p] = st.session_state[f"ui_prov_{p}"]

    def select_all_provs():
        for p in provinces:
            st.session_state.prov_selections[p] = True
            if f"ui_prov_{p}" in st.session_state:
                st.session_state[f"ui_prov_{p}"] = True

    def deselect_all_provs():
        for p in provinces:
            st.session_state.prov_selections[p] = False
            if f"ui_prov_{p}" in st.session_state:
                st.session_state[f"ui_prov_{p}"] = False

    with st.expander("Pilih Provinsi"):
        search_prov = st.text_input("Cari provinsi").lower()

        col1, col2 = st.columns(2)
        col1.button("Pilih semua", on_click=select_all_provs, key="btn_all_prov")
        col2.button("Batal semua", on_click=deselect_all_provs, key="btn_none_prov")

        cols_prov = st.columns(2)
        visible_prov_count = 0
        for p in provinces:
            if search_prov in p.lower():
                cols_prov[visible_prov_count % 2].checkbox(
                    p,
                    value=st.session_state.prov_selections.get(p, True),
                    key=f"ui_prov_{p}",
                    on_change=update_prov,
                    args=(p,),
                )
                visible_prov_count += 1

    selected_provinces = [
        p for p in provinces if st.session_state.prov_selections.get(p, True)
    ]

    # A. national_year_df
    national_year_df = df[df["tahun"] == selected_year].copy()

    # Filter available kab/kota for selected year and provinces
    kabupaten_scope_df = national_year_df[
        national_year_df["provinsi"].isin(selected_provinces)
    ]
    kabupaten_display_df = (
        kabupaten_scope_df[["kabupaten_kota", "provinsi"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["kabupaten_kota", "provinsi"])
    )

    kabupaten_lookup = {
        f"{row.kabupaten_kota} — {row.provinsi}": (row.kabupaten_kota, row.provinsi)
        for row in kabupaten_display_df.itertuples(index=False)
    }
    kab_labels = list(kabupaten_lookup.keys())

    if "kab_selections" not in st.session_state:
        st.session_state.kab_selections = {}

    st.session_state.kab_selections = {
        k: v for k, v in st.session_state.kab_selections.items() if k in kab_labels
    }

    for k in kab_labels:
        if k not in st.session_state.kab_selections:
            st.session_state.kab_selections[k] = True

    def update_kab(k):
        st.session_state.kab_selections[k] = st.session_state[f"ui_kab_{k}"]

    def select_all_kabs():
        for k in kab_labels:
            st.session_state.kab_selections[k] = True
            if f"ui_kab_{k}" in st.session_state:
                st.session_state[f"ui_kab_{k}"] = True

    def deselect_all_kabs():
        for k in kab_labels:
            st.session_state.kab_selections[k] = False
            if f"ui_kab_{k}" in st.session_state:
                st.session_state[f"ui_kab_{k}"] = False

    with st.expander("Pilih Kabupaten/Kota"):
        search_kab = st.text_input("Cari kabupaten/kota").lower()

        col1, col2 = st.columns(2)
        col1.button("Pilih semua", on_click=select_all_kabs, key="btn_all_kab")
        col2.button("Batal semua", on_click=deselect_all_kabs, key="btn_none_kab")

        cols_kab = st.columns(2)
        visible_kab_count = 0
        for k in kab_labels:
            if search_kab in k.lower():
                cols_kab[visible_kab_count % 2].checkbox(
                    k,
                    value=st.session_state.kab_selections.get(k, True),
                    key=f"ui_kab_{k}",
                    on_change=update_kab,
                    args=(k,),
                )
                visible_kab_count += 1

    selected_kabupaten_labels = [
        k for k in kab_labels if st.session_state.kab_selections.get(k, False)
    ]
    selected_kabupaten_pairs = [kabupaten_lookup[k] for k in selected_kabupaten_labels]


# -------------------------------------------------------------------
# Filtered datasets (Logical Scopes)
# -------------------------------------------------------------------

# B. detail_year_df
detail_year_df = kabupaten_scope_df.copy()
if selected_kabupaten_pairs:
    idx_detail = pd.MultiIndex.from_frame(
        detail_year_df[["kabupaten_kota", "provinsi"]]
    )
    detail_year_df = detail_year_df[idx_detail.isin(selected_kabupaten_pairs)].copy()
else:
    detail_year_df = pd.DataFrame(columns=df.columns)

# D. historical_province_df
if selected_provinces:
    historical_province_df = df[df["provinsi"].isin(selected_provinces)].copy()
else:
    historical_province_df = pd.DataFrame(columns=df.columns)

# C. historical_detail_df
historical_detail_df = historical_province_df.copy()
if selected_kabupaten_pairs:
    idx_hist_detail = pd.MultiIndex.from_frame(
        historical_detail_df[["kabupaten_kota", "provinsi"]]
    )
    historical_detail_df = historical_detail_df[
        idx_hist_detail.isin(selected_kabupaten_pairs)
    ].copy()
else:
    historical_detail_df = pd.DataFrame(columns=df.columns)


is_all_indonesia = (len(selected_provinces) == len(provinces)) and (
    len(selected_kabupaten_pairs) == len(kab_labels)
)


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

st.markdown(
    """
    <div class="dashboard-title">Dashboard Pengelolaan Sampah Indonesia</div>
    <div class="dashboard-subtitle">
        Dashboard interaktif untuk menganalisis timbulan sampah, kinerja pengelolaan,
        komposisi sampah, peringkat regional, dan peta sebaran tingkat provinsi.
    </div>
    """,
    unsafe_allow_html=True,
)

if not detail_year_df.empty:
    comp_terkelola = (
        detail_year_df["terkelola_ton"].notna().sum() / len(detail_year_df) * 100
    )
    if comp_terkelola < 100:
        st.warning(
            f"Data pada tahun {selected_year} untuk daerah yang dipilih belum lengkap (Cakupan data sampah terkelola: {comp_terkelola:.1f}%). "
            f"Beberapa nilai bersifat parsial atau belum tersedia."
        )

if not selected_provinces or not selected_kabupaten_pairs:
    st.warning("Tidak ada provinsi atau kabupaten/kota yang dipilih pada filter.")


# -------------------------------------------------------------------
# KPI cards
# -------------------------------------------------------------------

if detail_year_df.empty:
    total_generated, total_managed, unmanaged_gap, management_rate = (
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    )
else:
    total_generated = safe_sum(detail_year_df["timbulan_tahunan_ton"])
    total_managed = safe_sum(detail_year_df["terkelola_ton"])
    if pd.notna(total_generated) and pd.notna(total_managed):
        unmanaged_gap = total_generated - total_managed
    else:
        unmanaged_gap = safe_sum(detail_year_df["gap_tidak_tertangani_ton"])

    management_rate = calculate_management_rate(total_generated, total_managed)

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

kpi_col1.metric(
    "Total timbulan sampah",
    format_tons(total_generated),
    help="Jumlah timbulan sampah tahunan berdasarkan filter yang dipilih.",
)
kpi_col2.metric(
    "Total sampah terkelola",
    format_tons(total_managed),
    help="Jumlah sampah yang berhasil dikelola berdasarkan filter yang dipilih.",
)
kpi_col3.metric(
    "Gap tidak tertangani",
    format_tons(unmanaged_gap),
    help="Dihitung dari total timbulan sampah dikurangi total sampah terkelola (atau menggunakan field data jika tersedia).",
)
kpi_col4.metric(
    "Persentase pengelolaan",
    format_percent(management_rate),
    help="Dihitung dari total sampah terkelola dibagi total timbulan sampah.",
)

if is_all_indonesia:
    scope_location = "Seluruh Indonesia"
elif len(selected_kabupaten_pairs) == len(kab_labels) and len(kab_labels) > 0:
    scope_location = ", ".join(selected_provinces)
else:
    scope_location = f"{len(selected_kabupaten_pairs)} Kabupaten/Kota"

st.markdown(
    f"""
    <div class="note-box">
    <b>Cakupan saat ini:</b> {selected_year} · {scope_location} ·
    {format_number(len(detail_year_df))} data kabupaten/kota ·
    {format_number(detail_year_df["provinsi"].nunique())} provinsi.
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Geospatial analysis
# -------------------------------------------------------------------

st.markdown(
    '<div class="section-title">Analisis Geospasial</div>', unsafe_allow_html=True
)

map_filter_col, map_note_col = st.columns([1.2, 3])

with map_filter_col:
    selected_map_metric_label = st.selectbox(
        "Metrik peta",
        options=list(MAP_METRICS.keys()),
        index=0,
        key="map_metric_selectbox",
    )

province_df = aggregate_province(national_year_df)
map_metric = MAP_METRICS[selected_map_metric_label]

geojson = load_geojson()

if geojson is None:
    st.info("File GeoJSON belum ditemukan. Peta choropleth tidak dapat ditampilkan.")
    st.plotly_chart(
        make_empty_figure(
            "GeoJSON provinsi Indonesia dibutuhkan untuk menampilkan peta choropleth."
        ),
        width="stretch",
    )
elif province_df.empty:
    st.plotly_chart(
        make_empty_figure("Data nasional tahun ini tidak tersedia."), width="stretch"
    )
else:
    csv_norm_names = set(province_df["prov_norm"].dropna())
    geojson, geo_property, unmatched = prepare_geojson(geojson, csv_norm_names)

    if geo_property is None:
        st.error("Properti nama provinsi di GeoJSON tidak dapat dideteksi.")
        st.plotly_chart(
            make_empty_figure(
                "Properti nama provinsi di GeoJSON tidak dapat dideteksi."
            ),
            width="stretch",
        )
    else:
        map_fig = px.choropleth_map(
            province_df,
            geojson=geojson,
            locations="prov_norm",
            color=map_metric,
            hover_name="provinsi",
            hover_data={
                "prov_norm": False,
                "timbulan_tahunan_ton": ":,.0f",
                "terkelola_ton": ":,.0f",
                "gap_tidak_tertangani_ton": ":,.0f",
                "management_rate": ":.2f",
                "total_kabupaten_kota": True,
            },
            labels={
                "timbulan_tahunan_ton": "Timbulan sampah tahunan",
                "terkelola_ton": "Sampah terkelola",
                "gap_tidak_tertangani_ton": "Gap tidak tertangani",
                "management_rate": "Persentase pengelolaan (%)",
                "total_kabupaten_kota": "Jumlah kabupaten/kota",
            },
            color_continuous_scale=MAP_COLOR_SCALE,
            opacity=0.90,
            center={"lat": -2.5, "lon": 118},
            zoom=3.35,
            map_style="white-bg",
        )

        map_fig.update_traces(
            marker_line_width=0.7,
            marker_line_color="rgba(15, 23, 42, 0.65)",
        )

        province_df["is_selected"] = province_df["provinsi"].isin(selected_provinces)
        selected_provs = province_df[province_df["is_selected"]]

        if not selected_provs.empty and len(selected_provs) < len(province_df):
            map_fig.add_trace(
                go.Choroplethmap(
                    geojson=geojson,
                    locations=selected_provs["prov_norm"],
                    z=[1] * len(selected_provs),
                    colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                    showscale=False,
                    marker_line_color="#f59e0b",
                    marker_line_width=3,
                    hoverinfo="skip",
                )
            )
            st.caption(
                "Peta menampilkan konteks nasional. Garis batas oranye menandakan provinsi yang dipilih pada filter."
            )

        map_fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=650,
            margin=dict(l=0, r=0, t=20, b=0),
            font=dict(color=TEXT_COLOR),
            coloraxis_colorbar=dict(
                title=dict(text=selected_map_metric_label, font=dict(color=TEXT_COLOR)),
                tickfont=dict(color=TEXT_COLOR),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(map_fig, width="stretch")


# -------------------------------------------------------------------
# 1. Bar chart — Timbulan per provinsi top 15
# -------------------------------------------------------------------

st.markdown(
    '<div class="section-title">Timbulan Sampah per Provinsi</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Siapa penyumbang sampah terbesar? Grafik ini menampilkan 15 provinsi dengan timbulan sampah tahunan tertinggi secara nasional."
)

nat_timbulan = aggregate_province(national_year_df)
if nat_timbulan.empty:
    st.plotly_chart(make_empty_figure("Data nasional tidak tersedia."), width="stretch")
else:
    top_timbulan_df = nat_timbulan.sort_values(
        "timbulan_tahunan_ton", ascending=False
    ).head(15)
    top_timbulan_df = top_timbulan_df.sort_values(
        "timbulan_tahunan_ton", ascending=True
    )

    if not is_all_indonesia:
        top_timbulan_df["color"] = np.where(
            top_timbulan_df["provinsi"].isin(selected_provinces), "#f59e0b", "#2563eb"
        )
    else:
        top_timbulan_df["color"] = "#2563eb"

    top_timbulan_df["label_timbulan"] = top_timbulan_df["timbulan_tahunan_ton"].apply(
        lambda x: (
            f"{x / 1_000_000:.2f}M"
            if pd.notna(x) and x >= 1_000_000
            else (f"{x / 1_000:.0f}K" if pd.notna(x) else "N/A")
        )
    )

    fig_timbulan = px.bar(
        top_timbulan_df,
        x="timbulan_tahunan_ton",
        y="provinsi",
        orientation="h",
        text="label_timbulan",
        hover_data={
            "timbulan_tahunan_ton": ":,.0f",
            "terkelola_ton": ":,.0f",
            "gap_tidak_tertangani_ton": ":,.0f",
            "management_rate": ":.2f",
            "total_kabupaten_kota": True,
        },
        labels={
            "timbulan_tahunan_ton": "Timbulan sampah (ton/tahun)",
            "provinsi": "",
            "terkelola_ton": "Sampah terkelola",
            "gap_tidak_tertangani_ton": "Gap tidak tertangani",
            "management_rate": "Persentase pengelolaan (%)",
            "total_kabupaten_kota": "Jumlah kabupaten/kota",
        },
    )

    fig_timbulan.update_traces(
        marker_color=top_timbulan_df["color"],
        marker_line_color="#1e40af",
        marker_line_width=1,
        textposition="outside",
        cliponaxis=False,
    )

    fig_timbulan.update_layout(
        height=560,
        margin=dict(l=10, r=60, t=25, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(title=""),
        showlegend=False,
    )

    st.plotly_chart(fig_timbulan, width="stretch", config={"displayModeBar": False})


# -------------------------------------------------------------------
# 2. Pie chart — Kategori pengelolaan
# -------------------------------------------------------------------

st.markdown(
    '<div class="section-title">Kategori Pengelolaan Sampah</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Secara keseluruhan, kondisi pengelolaan sampah untuk daerah yang dipilih itu baik atau buruk?"
)

category_counts = (
    detail_year_df["kategori_pengelolaan"]
    .dropna()
    .value_counts()
    .reindex(CATEGORY_ORDER)
    .dropna()
    .reset_index()
)
category_counts.columns = ["Kategori", "Jumlah Daerah"]

if category_counts.empty:
    st.plotly_chart(
        make_empty_figure("Data kategori pengelolaan tidak tersedia."), width="stretch"
    )
else:
    fig_category = px.pie(
        category_counts,
        names="Kategori",
        values="Jumlah Daerah",
        hole=0.45,
        color="Kategori",
        color_discrete_map=CATEGORY_COLORS,
    )

    fig_category.update_traces(
        texttemplate="%{label}<br>%{percent}",
        textposition="inside",
        marker=dict(line=dict(color="#f8fafc", width=2)),
    )

    fig_category.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=25, b=10),
        showlegend=True,
        legend_title_text="",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    apply_chart_theme(fig_category)
    st.plotly_chart(fig_category, width="stretch", config={"displayModeBar": False})

# -------------------------------------------------------------------
# 4. Line chart — Tren % terkelola 2018–2025
# -------------------------------------------------------------------

min_yr = historical_detail_df["tahun"].min() if not historical_detail_df.empty else 2018
max_yr = historical_detail_df["tahun"].max() if not historical_detail_df.empty else 2025

st.markdown(
    f'<div class="section-title">Tren Persentase Sampah Terkelola {min_yr}–{max_yr}</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Menampilkan rata-rata persentase berdasarkan timbulan dan sampah terkelola di wilayah yang dipilih per tahun."
)

trend_valid = historical_detail_df.dropna(subset=["terkelola_ton"]).copy()
if trend_valid.empty:
    st.plotly_chart(
        make_empty_figure("Data tren persentase pengelolaan tidak tersedia."),
        width="stretch",
    )
else:
    trend_grouped = (
        trend_valid.groupby("tahun", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "timbulan": g["timbulan_tahunan_ton"].sum(min_count=1),
                    "terkelola": g["terkelola_ton"].sum(min_count=1),
                }
            )
        )
        .reset_index()
    )

    trend_grouped["pct_terkelola"] = np.where(
        (trend_grouped["timbulan"] > 0) & (trend_grouped["terkelola"].notna()),
        trend_grouped["terkelola"] / trend_grouped["timbulan"] * 100,
        np.nan,
    )
    trend_grouped = trend_grouped.dropna(subset=["pct_terkelola"]).sort_values("tahun")

    if trend_grouped.empty:
        st.plotly_chart(
            make_empty_figure(
                "Data tren persentase pengelolaan tidak tersedia setelah agregasi."
            ),
            width="stretch",
        )
    else:
        fig_trend_managed = px.line(
            trend_grouped,
            x="tahun",
            y="pct_terkelola",
            markers=True,
            labels={
                "tahun": "Tahun",
                "pct_terkelola": "Rata-rata persentase terkelola (%)",
            },
            hover_data={"pct_terkelola": ":.2f"},
        )

        fig_trend_managed.update_traces(
            line=dict(color="#16a34a", width=3), marker=dict(size=8, color="#16a34a")
        )
        fig_trend_managed.update_layout(
            height=460,
            margin=dict(l=10, r=10, t=25, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(dtick=1, gridcolor=GRID_COLOR, zeroline=False),
            yaxis=dict(
                gridcolor=GRID_COLOR,
                zeroline=False,
                range=[0, max(100, trend_grouped["pct_terkelola"].max() + 5)],
            ),
            showlegend=False,
        )
        st.plotly_chart(
            fig_trend_managed, width="stretch", config={"displayModeBar": False}
        )


# -------------------------------------------------------------------
# 5. Scatter plot — Timbulan vs % terkelola
# -------------------------------------------------------------------

st.markdown(
    '<div class="section-title">Hubungan Timbulan Sampah dan Persentase Pengelolaan</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Apakah daerah dengan timbulan sampah lebih besar justru lebih buruk pengelolaannya?"
)

scatter_df = detail_year_df.dropna(
    subset=["timbulan_tahunan_ton", "pct_terkelola"]
).copy()

if scatter_df.empty:
    st.plotly_chart(
        make_empty_figure("Data scatter plot tidak tersedia."), width="stretch"
    )
else:
    if (
        len(scatter_df) >= 2
        and scatter_df["timbulan_tahunan_ton"].nunique() > 1
        and scatter_df["pct_terkelola"].nunique() > 1
    ):
        corr_val = scatter_df["timbulan_tahunan_ton"].corr(scatter_df["pct_terkelola"])
        direction = "positif" if corr_val > 0 else "negatif"
        interpretation = interpret_correlation(corr_val)
        corr_text = f"Koefisien korelasi Pearson: **{corr_val:.2f}** ({interpretation}, arah: {direction}). *Korelasi tidak membuktikan sebab-akibat.*"
    else:
        corr_text = "Data tidak cukup atau kurang bervariasi untuk menghitung korelasi."

    fig_scatter = px.scatter(
        scatter_df,
        x="timbulan_tahunan_ton",
        y="pct_terkelola",
        color="kategori_pengelolaan",
        color_discrete_map=CATEGORY_COLORS,
        hover_name="kabupaten_kota",
        hover_data={
            "provinsi": True,
            "timbulan_tahunan_ton": ":,.0f",
            "terkelola_ton": ":,.0f",
            "gap_tidak_tertangani_ton": ":,.0f",
            "pct_terkelola": ":.2f",
        },
        labels={
            "timbulan_tahunan_ton": "Timbulan sampah (ton/tahun)",
            "pct_terkelola": "Persentase terkelola (%)",
            "kategori_pengelolaan": "Kategori pengelolaan",
        },
    )

    fig_scatter.update_traces(
        marker=dict(
            size=10,
            line=dict(width=0.8, color="#0f172a"),
            opacity=0.78,
        )
    )

    fig_scatter.update_layout(
        height=540,
        margin=dict(l=10, r=10, t=25, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
    )
    st.plotly_chart(fig_scatter, width="stretch", config={"displayModeBar": False})


# -------------------------------------------------------------------
# 6. Perbandingan Kabupaten/Kota
# -------------------------------------------------------------------

st.markdown(
    f'<div class="section-title">Perbandingan Kabupaten/Kota Tahun {selected_year}</div>',
    unsafe_allow_html=True,
)
st.caption("Membandingkan metrik spesifik antar kota pada tahun yang dipilih.")

comp_metrics = {
    "Timbulan sampah tahunan (ton)": "timbulan_tahunan_ton",
    "Persentase pengelolaan (%)": "pct_terkelola",
    "Sampah terkelola (ton)": "terkelola_ton",
    "Gap tidak tertangani (ton)": "gap_tidak_tertangani_ton",
    "Recycling rate (%)": "recycling_rate",
}

c1, c2, c3 = st.columns(3)
comp_metric_label = c1.selectbox("Metrik", list(comp_metrics.keys()), key="comp_metric")
comp_mode = c2.selectbox("Urutkan", ["Tertinggi", "Terendah"], key="comp_mode")
comp_count = c3.selectbox(
    "Jumlah tampilan", [10, 15, 20, 30], index=1, key="comp_count"
)

comp_col = comp_metrics[comp_metric_label]

if detail_year_df.empty:
    st.plotly_chart(make_empty_figure("Data detail tidak tersedia."), width="stretch")
else:
    comp_df = detail_year_df.dropna(subset=[comp_col]).copy()
    if comp_df.empty:
        st.plotly_chart(
            make_empty_figure("Data tidak tersedia untuk metrik yang dipilih."),
            width="stretch",
        )
    else:
        if len(comp_df) <= comp_count:
            display_comp = comp_df
        else:
            display_comp = select_ranked_rows(comp_df, comp_col, comp_mode, comp_count)

        display_comp = display_comp.sort_values(comp_col, ascending=True)

        is_pct = "%)" in comp_metric_label
        fmt = ":.2f" if is_pct else ":,.0f"

        fig_comp = px.bar(
            display_comp,
            x=comp_col,
            y="kabupaten_kota",
            orientation="h",
            hover_name="kabupaten_kota",
            hover_data={"provinsi": True, comp_col: fmt, "kabupaten_kota": False},
            labels={
                comp_col: comp_metric_label,
                "kabupaten_kota": "Kabupaten/Kota",
                "provinsi": "Provinsi",
            },
        )

        fig_comp.update_traces(marker_color="#0ea5e9")
        fig_comp.update_layout(
            height=max(400, len(display_comp) * 25),
            margin=dict(l=10, r=10, t=25, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
            yaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        )
        st.plotly_chart(fig_comp, width="stretch", config={"displayModeBar": False})


# -------------------------------------------------------------------
# 7. Horizontal bar — Komposisi jenis sampah
# -------------------------------------------------------------------

st.markdown(
    '<div class="section-title">Komposisi Jenis Sampah</div>', unsafe_allow_html=True
)
st.caption("Jenis sampah apa yang paling dominan pada wilayah yang dipilih?")

existing_composition_cols = [
    col for col in COMPOSITION_COLUMNS if col in detail_year_df.columns
]
composition_source = (
    detail_year_df[existing_composition_cols + ["timbulan_tahunan_ton"]]
    .dropna(subset=["timbulan_tahunan_ton"])
    .copy()
)

weighted_values = {}
for col in existing_composition_cols:
    valid_rows = composition_source.dropna(subset=[col])
    if valid_rows.empty:
        continue
    total_weight = valid_rows["timbulan_tahunan_ton"].sum()
    if total_weight <= 0:
        weighted_values[col] = valid_rows[col].mean()
    else:
        weighted_values[col] = (
            valid_rows[col] * valid_rows["timbulan_tahunan_ton"]
        ).sum() / total_weight

composition = pd.Series(weighted_values).dropna()
composition = composition[composition > 0]

if composition.empty:
    st.plotly_chart(
        make_empty_figure("Data komposisi sampah tidak tersedia."), width="stretch"
    )
else:
    composition_df = pd.DataFrame(
        {
            "Jenis Sampah": [COMPOSITION_COLUMNS[col] for col in composition.index],
            "Rata-rata Persentase": composition.values,
        }
    ).sort_values("Rata-rata Persentase", ascending=True)

    composition_df["Label"] = composition_df["Rata-rata Persentase"].apply(
        lambda x: f"{x:.1f}%"
    )

    fig_composition_bar = px.bar(
        composition_df,
        x="Rata-rata Persentase",
        y="Jenis Sampah",
        orientation="h",
        text="Label",
        color="Rata-rata Persentase",
        color_continuous_scale=[
            [0.0, "#dbeafe"],
            [0.4, "#38bdf8"],
            [0.7, "#2563eb"],
            [1.0, "#1e3a8a"],
        ],
        hover_data={"Rata-rata Persentase": ":.2f"},
        labels={"Rata-rata Persentase": "Rata-rata komposisi (%)", "Jenis Sampah": ""},
    )

    fig_composition_bar.update_traces(
        textposition="outside",
        marker_line_color="#1e3a8a",
        marker_line_width=0.8,
        cliponaxis=False,
    )

    fig_composition_bar.update_layout(
        height=500,
        margin=dict(l=10, r=70, t=25, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            gridcolor=GRID_COLOR, zeroline=False, title="Rata-rata komposisi (%)"
        ),
        yaxis=dict(title=""),
        coloraxis_showscale=False,
        showlegend=False,
    )
    st.plotly_chart(
        fig_composition_bar, width="stretch", config={"displayModeBar": False}
    )


# -------------------------------------------------------------------
# 8. Heatmap — % terkelola provinsi
# -------------------------------------------------------------------

st.markdown(
    f'<div class="section-title">Heatmap Persentase Pengelolaan per Provinsi ({min_yr}–{max_yr})</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Provinsi mana yang konsisten bagus, dan mana yang konsisten buruk selama bertahun-tahun? (Skala provinsi)"
)

if historical_province_df.empty:
    st.plotly_chart(
        make_empty_figure("Data heatmap persentase pengelolaan tidak tersedia."),
        width="stretch",
    )
else:
    hist_prov_agg = (
        historical_province_df.groupby(["provinsi", "tahun"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "timbulan": g["timbulan_tahunan_ton"].sum(min_count=1),
                    "terkelola": g["terkelola_ton"].sum(min_count=1),
                }
            )
        )
        .reset_index()
    )

    hist_prov_agg["pct_terkelola"] = np.where(
        (hist_prov_agg["timbulan"] > 0) & (hist_prov_agg["terkelola"].notna()),
        hist_prov_agg["terkelola"] / hist_prov_agg["timbulan"] * 100,
        np.nan,
    )

    heatmap_managed = hist_prov_agg.dropna(subset=["pct_terkelola"])
    if heatmap_managed.empty:
        st.plotly_chart(
            make_empty_figure("Data heatmap persentase pengelolaan tidak tersedia."),
            width="stretch",
        )
    else:
        heatmap_managed = heatmap_managed.round(2).pivot(
            index="provinsi", columns="tahun", values="pct_terkelola"
        )

        fig_heatmap_managed = px.imshow(
            heatmap_managed,
            aspect="auto",
            color_continuous_scale=[
                [0.0, "#dc2626"],
                [0.35, "#f97316"],
                [0.65, "#facc15"],
                [1.0, "#16a34a"],
            ],
            zmin=0,
            zmax=100,
            text_auto=".1f",
            labels=dict(x="Tahun", y="Provinsi", color="% Terkelola"),
        )

        fig_heatmap_managed.update_layout(
            height=max(430, len(heatmap_managed) * 25),
            margin=dict(l=10, r=20, t=25, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_colorbar=dict(title=dict(text="% Terkelola")),
        )
        st.plotly_chart(
            fig_heatmap_managed, width="stretch", config={"displayModeBar": False}
        )


# -------------------------------------------------------------------
# 9. Stacked bar — Sampah tertangani vs tidak tertangani
# -------------------------------------------------------------------

st.markdown(
    '<div class="section-title">Sampah Tertangani vs Tidak Tertangani per Provinsi</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Dari total nasional, berapa yang sudah tertangani dan berapa yang masih menjadi gap tidak tertangani? Menampilkan Top 15 Nasional."
)

if nat_timbulan.empty:
    st.plotly_chart(make_empty_figure("Data nasional tidak tersedia."), width="stretch")
else:
    handled_gap_df = (
        nat_timbulan.sort_values("gap_tidak_tertangani_ton", ascending=False)
        .head(15)
        .copy()
    )

    handled_gap_df["terkelola_ton"] = handled_gap_df["terkelola_ton"].clip(lower=0)
    handled_gap_df["gap_tidak_tertangani_ton"] = handled_gap_df[
        "gap_tidak_tertangani_ton"
    ].clip(lower=0)
    handled_gap_df = handled_gap_df.sort_values(
        "gap_tidak_tertangani_ton", ascending=True
    )

    if not is_all_indonesia:
        handled_gap_df["is_selected"] = handled_gap_df["provinsi"].isin(
            selected_provinces
        )
    else:
        handled_gap_df["is_selected"] = False

    handled_gap_long = handled_gap_df.melt(
        id_vars=[
            "provinsi",
            "timbulan_tahunan_ton",
            "management_rate",
            "total_kabupaten_kota",
            "is_selected",
        ],
        value_vars=["terkelola_ton", "gap_tidak_tertangani_ton"],
        var_name="status",
        value_name="ton",
    )

    handled_gap_long["status"] = handled_gap_long["status"].map(
        {"terkelola_ton": "Tertangani", "gap_tidak_tertangani_ton": "Tidak tertangani"}
    )
    handled_gap_long["label_ton"] = handled_gap_long["ton"].apply(
        lambda x: (
            f"{x / 1_000_000:.2f}M"
            if pd.notna(x) and x >= 1_000_000
            else (f"{x / 1_000:.0f}K" if pd.notna(x) else "N/A")
        )
    )

    fig_handled_gap = px.bar(
        handled_gap_long,
        x="ton",
        y="provinsi",
        color="status",
        orientation="h",
        barmode="stack",
        text="label_ton",
        color_discrete_map={"Tertangani": "#16a34a", "Tidak tertangani": "#dc2626"},
        hover_data={
            "ton": ":,.0f",
            "timbulan_tahunan_ton": ":,.0f",
            "management_rate": ":.2f",
            "total_kabupaten_kota": True,
            "is_selected": False,
        },
        labels={
            "ton": "Ton/tahun",
            "provinsi": "",
            "status": "Status pengelolaan",
            "timbulan_tahunan_ton": "Total timbulan sampah",
            "management_rate": "Persentase pengelolaan (%)",
            "total_kabupaten_kota": "Jumlah kabupaten/kota",
        },
    )

    # optionally highlight selected provinces if they happen to be in the Top 15. Plotly stacked bar outline is standard, we won't colorize lines differently per bar to keep it simple, or we can just let it be.
    fig_handled_gap.update_traces(
        marker_line_color="#0f172a",
        marker_line_width=0.7,
        textposition="inside",
        insidetextanchor="middle",
        cliponaxis=False,
    )

    fig_handled_gap.update_layout(
        height=560,
        margin=dict(l=10, r=40, t=25, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            gridcolor=GRID_COLOR, zeroline=False, title="Jumlah sampah (ton/tahun)"
        ),
        yaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
    )
    st.plotly_chart(fig_handled_gap, width="stretch", config={"displayModeBar": False})


# -------------------------------------------------------------------
# 10. Grouped bar — Pengurangan vs penanganan vs daur ulang
# -------------------------------------------------------------------

st.markdown(
    '<div class="section-title">Pengurangan, Penanganan, dan Daur Ulang per Provinsi</div>',
    unsafe_allow_html=True,
)
st.caption("Menampilkan Top 15 Nasional untuk metrik detail pengelolaan.")

if nat_timbulan.empty:
    st.plotly_chart(make_empty_figure("Data nasional tidak tersedia."), width="stretch")
else:
    grouped_metrics_df = (
        nat_timbulan.sort_values("penanganan_ton", ascending=False).head(15).copy()
    )

    grouped_long = grouped_metrics_df.melt(
        id_vars=["provinsi"],
        value_vars=["pengurangan_ton", "penanganan_ton", "daur_ulang_ton"],
        var_name="metrik",
        value_name="ton",
    )

    grouped_long["metrik"] = grouped_long["metrik"].map(
        {
            "pengurangan_ton": "Pengurangan",
            "penanganan_ton": "Penanganan",
            "daur_ulang_ton": "Daur Ulang",
        }
    )

    fig_grouped = px.bar(
        grouped_long,
        x="provinsi",
        y="ton",
        color="metrik",
        barmode="group",
        color_discrete_map={
            "Pengurangan": "#0ea5e9",
            "Penanganan": "#16a34a",
            "Daur Ulang": "#f97316",
        },
        hover_data={"ton": ":,.0f"},
        labels={"provinsi": "Provinsi", "ton": "Ton/tahun", "metrik": "Metrik"},
    )

    fig_grouped.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=25, b=100),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickangle=-35, gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
    )
    st.plotly_chart(fig_grouped, width="stretch", config={"displayModeBar": False})


# -------------------------------------------------------------------
# 11. Scatter plot — % pengurangan vs % penanganan per kab/kota
# -------------------------------------------------------------------

st.markdown(
    '<div class="section-title">Hubungan Persentase Pengurangan dan Persentase Penanganan</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Setiap titik mewakili satu kabupaten/kota. Sumbu X menunjukkan persentase pengurangan sampah, sedangkan sumbu Y menunjukkan persentase penanganan sampah berdasarkan tahun dan wilayah yang dipilih."
)

scatter_reduce_handle = detail_year_df.dropna(
    subset=["pct_pengurangan", "pct_penanganan"]
).copy()

if scatter_reduce_handle.empty:
    st.plotly_chart(
        make_empty_figure("Data pengurangan dan penanganan tidak tersedia."),
        width="stretch",
    )
else:
    fig_reduce_handle = px.scatter(
        scatter_reduce_handle,
        x="pct_pengurangan",
        y="pct_penanganan",
        color="provinsi",
        hover_name="kabupaten_kota",
        hover_data={
            "provinsi": True,
            "pct_pengurangan": ":.2f",
            "pct_penanganan": ":.2f",
            "pct_terkelola": ":.2f",
        },
        labels={
            "pct_pengurangan": "Persentase pengurangan (%)",
            "pct_penanganan": "Persentase penanganan (%)",
            "pct_terkelola": "Persentase terkelola (%)",
            "provinsi": "Provinsi",
        },
    )

    fig_reduce_handle.update_traces(
        marker=dict(size=9, line=dict(width=0.7, color="#0f172a"), opacity=0.78)
    )
    fig_reduce_handle.update_layout(
        height=540,
        margin=dict(l=10, r=10, t=25, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        showlegend=False,
    )
    st.plotly_chart(
        fig_reduce_handle, width="stretch", config={"displayModeBar": False}
    )


# -------------------------------------------------------------------
# 12. Line chart — Tren recycling rate
# -------------------------------------------------------------------

st.markdown(
    f'<div class="section-title">Tren Recycling Rate {min_yr}–{max_yr}</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Recycling rate memiliki beberapa outlier, sehingga grafik ini digunakan sebagai indikator pendukung."
)

trend_recycling_valid = historical_detail_df.dropna(subset=["recycling_rate"])

if trend_recycling_valid.empty:
    st.plotly_chart(
        make_empty_figure("Data tren recycling rate tidak tersedia."), width="stretch"
    )
else:
    trend_recycling = (
        trend_recycling_valid.groupby("tahun", as_index=False)["recycling_rate"]
        .mean()
        .sort_values("tahun")
    )

    fig_trend_recycling = px.line(
        trend_recycling,
        x="tahun",
        y="recycling_rate",
        markers=True,
        labels={"tahun": "Tahun", "recycling_rate": "Rata-rata recycling rate (%)"},
        hover_data={"recycling_rate": ":.2f"},
    )

    fig_trend_recycling.update_traces(
        line=dict(color="#0ea5e9", width=3), marker=dict(size=8, color="#0ea5e9")
    )
    fig_trend_recycling.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=25, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(dtick=1, gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        showlegend=False,
    )
    st.plotly_chart(
        fig_trend_recycling, width="stretch", config={"displayModeBar": False}
    )


# -------------------------------------------------------------------
# 13. Heatmap recycling rate provinsi
# -------------------------------------------------------------------

st.markdown(
    f'<div class="section-title">Heatmap Recycling Rate per Provinsi ({min_yr}–{max_yr})</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Warna menunjukkan recycling rate setiap provinsi pada masing-masing tahun. Sel kosong menandakan data tidak tersedia, bukan bernilai 0%."
)

if historical_province_df.empty:
    st.plotly_chart(
        make_empty_figure("Data heatmap recycling rate tidak tersedia."),
        width="stretch",
    )
else:
    recycling_heatmap = (
        historical_province_df.dropna(subset=["recycling_rate"])
        .groupby(["provinsi", "tahun"], as_index=False)["recycling_rate"]
        .mean()
        .round(2)
        .pivot(index="provinsi", columns="tahun", values="recycling_rate")
    )

    if recycling_heatmap.empty:
        st.plotly_chart(
            make_empty_figure("Data heatmap recycling rate tidak tersedia."),
            width="stretch",
        )
    else:
        fig_heatmap_recycling = px.imshow(
            recycling_heatmap,
            aspect="auto",
            color_continuous_scale="Blues",
            zmin=0,
            zmax=100,
            text_auto=".1f",
            labels=dict(x="Tahun", y="Provinsi", color="Recycling rate (%)"),
        )

        fig_heatmap_recycling.update_layout(
            height=max(430, len(recycling_heatmap) * 25),
            margin=dict(l=10, r=20, t=25, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_colorbar=dict(title=dict(text="Recycling rate (%)")),
        )
        st.plotly_chart(
            fig_heatmap_recycling, width="stretch", config={"displayModeBar": False}
        )
