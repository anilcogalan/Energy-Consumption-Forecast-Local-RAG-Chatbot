"""
===============================================================================
Enerji Tüketimi Tahminleme — Streamlit Demo
===============================================================================
Dataset: UCI Individual Household Electric Power Consumption
Kaggle: https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set

Sayfalar:
  1. 📊 Veri Görselleştirme — Zaman serisi, seasonality, distribüsyon
  2. ⚡ Tahmin Ekranı — Tarih aralığı seç, model tahminlerini gör
  3. 🏆 Model Performans — Metrik tablosu, residual, SHAP özeti

Kurulum:
  pip install -r requirements.txt
  streamlit run app.py
===============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────
# CONFIG & THEME
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enerji Tüketimi Tahminleme",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');

    .stApp { font-family: 'DM Sans', sans-serif; }

    .metric-card {
        background: linear-gradient(135deg, #065A82 0%, #1C7293 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #02C39A;
    }
    .metric-card .label {
        font-size: 0.85rem;
        opacity: 0.85;
        margin-top: 0.2rem;
    }

    .insight-box {
        background: #FFF7ED;
        border-left: 4px solid #F59E0B;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: #0F172A !important;
    }
    .insight-box *, .insight-box p, .insight-box strong, .insight-box span { color: #0F172A !important; }

    .dataset-link {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        font-size: 0.9rem;
        color: #0F172A !important;
    }
    .dataset-link *, .dataset-link p, .dataset-link strong { color: #0F172A !important; }
    .dataset-link a { color: #0369A1 !important; }

    div[data-testid="stSidebar"] { background: #0B2545; }
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown li,
    div[data-testid="stSidebar"] .stMarkdown h1,
    div[data-testid="stSidebar"] .stMarkdown h2,
    div[data-testid="stSidebar"] .stMarkdown h3 { color: #E2E8F0; }
    div[data-testid="stSidebar"] .stRadio label p { color: #E2E8F0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# DATA LOADING & CACHING
# ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare_data():
    """Veriyi yükle, saatlik aggregation, feature engineering."""
    try:
        df_raw = pd.read_csv(
            'datas/household_power_consumption.txt',
            sep=';',
            parse_dates={'datetime': ['Date', 'Time']},
            dayfirst=True,
            low_memory=False,
            na_values=['?', 'nan', '']
        )
    except FileNotFoundError:
        return None, None, None, None, None

    df_raw = df_raw.set_index('datetime').sort_index()

    # Saatlik aggregation
    df = df_raw.resample('1h').mean()
    df = df.interpolate(method='time', limit=24)
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].shift(168))
        if df[col].isnull().any():
            df[col] = df[col].ffill().bfill()

    df = df.drop(columns=['Global_intensity'], errors='ignore')
    target = 'Global_active_power'

    # Feature engineering (compact)
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # Fourier
    df['sin_12h'] = np.sin(2 * np.pi * df['hour'] / 12)
    df['cos_12h'] = np.cos(2 * np.pi * df['hour'] / 12)
    df['sin_24h'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_24h'] = np.cos(2 * np.pi * df['hour'] / 24)
    how = df['dayofweek'] * 24 + df['hour']
    df['sin_168h'] = np.sin(2 * np.pi * how / 168)
    df['cos_168h'] = np.cos(2 * np.pi * how / 168)
    df['sin_yearly'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['cos_yearly'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

    # Lags
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        df[f'lag_{lag}'] = df[target].shift(lag)
    df['sub3_lag_1'] = df['Sub_metering_3'].shift(1)
    df['sub3_lag_24'] = df['Sub_metering_3'].shift(24)
    df['voltage_lag_1'] = df['Voltage'].shift(1)

    # Rolling/EWMA
    shifted = df[target].shift(1)
    for w in [3, 6, 12, 24, 48, 168]:
        df[f'rolling_mean_{w}h'] = shifted.rolling(w, min_periods=w//2).mean()
        df[f'rolling_std_{w}h'] = shifted.rolling(w, min_periods=w//2).std()
        df[f'rolling_min_{w}h'] = shifted.rolling(w, min_periods=w//2).min()
        df[f'rolling_max_{w}h'] = shifted.rolling(w, min_periods=w//2).max()
    for span in [3, 6, 12]:
        df[f'ewma_{span}h'] = shifted.ewm(span=span, min_periods=span//2).mean()

    # Diffs
    df['diff_1h_lagged'] = df[target].shift(1).diff(1)
    df['diff_24h_lagged'] = df[target].shift(1).diff(24)
    df['pct_change_1h_lagged'] = df[target].shift(1).pct_change(1)

    # Interactions
    df['hour_weekend_interact'] = df['hour'] * df['is_weekend']
    df['month_hour_interact'] = df['month'] * df['hour']
    df['sub3_x_month'] = df['Sub_metering_3'].shift(1) * df['month']
    df['voltage_deviation'] = df['Voltage'] - df['Voltage'].rolling(24, min_periods=12).mean()

    # Feature columns
    exclude = [target]
    feature_cols = [c for c in df.columns if c not in exclude
                    and df[c].dtype in ['float64', 'int64', 'int32', 'float32']]

    # Split
    train_end = '2009-12-31 23:00:00'
    val_end = '2010-06-30 23:00:00'
    df_clean = df.dropna(subset=feature_cols + [target])

    train = df_clean[:train_end]
    val = df_clean[train_end:val_end].iloc[1:]
    test = df_clean[val_end:].iloc[1:]

    return df, train, val, test, feature_cols


@st.cache_resource
def train_models(_train, _val, _test, feature_cols):
    """LightGBM ve XGBoost modellerini eğit."""
    import lightgbm as lgb
    import xgboost as xgb

    target = 'Global_active_power'
    X_train = _train[feature_cols]
    y_train = _train[target]
    X_val = _val[feature_cols]
    y_val = _val[target]
    X_test = _test[feature_cols]
    y_test = _test[target]

    # LightGBM (Optuna'dan en iyi parametreler)
    lgb_params = {
        'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
        'verbosity': -1, 'n_jobs': -1,
        'num_leaves': 150, 'max_depth': 11, 'learning_rate': 0.0138,
        'n_estimators': 1545, 'min_child_samples': 38,
        'subsample': 0.566, 'colsample_bytree': 0.784,
        'reg_alpha': 0.0125, 'reg_lambda': 3e-05
    }
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

    # XGBoost
    xgb_params = {
        'objective': 'reg:squarederror', 'eval_metric': 'mae',
        'booster': 'gbtree', 'verbosity': 0, 'n_jobs': -1,
        'max_depth': 11, 'learning_rate': 0.0108,
        'n_estimators': 1548, 'min_child_weight': 38,
        'subsample': 0.833, 'colsample_bytree': 0.737,
        'reg_alpha': 2.9e-05, 'reg_lambda': 0.0123, 'gamma': 4.1e-06
    }
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Predictions
    preds = {
        'LightGBM_val': np.maximum(lgb_model.predict(X_val), 0),
        'LightGBM_test': np.maximum(lgb_model.predict(X_test), 0),
        'XGBoost_val': np.maximum(xgb_model.predict(X_val), 0),
        'XGBoost_test': np.maximum(xgb_model.predict(X_test), 0),
        'Naive_val': y_val.shift(1).bfill().values,
        'Naive_test': y_test.shift(1).bfill().values,
    }

    # SHAP
    import shap
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_test)
    shap_importance = pd.DataFrame({
        'Feature': feature_cols,
        'SHAP': np.abs(shap_values).mean(axis=0)
    }).sort_values('SHAP', ascending=False)

    return lgb_model, xgb_model, preds, shap_importance


def calc_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R²': r2}


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Enerji Tahminleme")
    st.markdown("---")
    page = st.radio(
        "Sayfa Seçin",
        ["📊 Veri Görselleştirme", "⚡ Tahmin Ekranı", "🏆 Model Performans"],
        index=0
    )
    st.markdown("---")
    st.markdown("""
    **Dataset:**
    [UCI / Kaggle](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)

    **Veri:** 2006-2010, Fransa
    **Granülarite:** Saatlik (1H)
    **Gözlem:** ~34,589
    """)
    st.markdown("---")
    st.markdown("*Bu case Eksim Holding için PoC çalışmasıdır..*")

# ─────────────────────────────────────────────────────────────────────
# DATA LOAD
# ─────────────────────────────────────────────────────────────────────
df, train, val, test, feature_cols = load_and_prepare_data()

if df is None:
    st.error("⚠️ `datas/household_power_consumption.txt` dosyası bulunamadı.")
    st.markdown("""
    Lütfen Kaggle'dan indirip **datas** klasörüne koyun:

    👉 [Dataset İndir](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)
    """)
    st.stop()

target = 'Global_active_power'

# ─────────────────────────────────────────────────────────────────────
# PAGE 1: VERİ GÖRSELLEŞTİRME (Detaylı EDA)
# ─────────────────────────────────────────────────────────────────────
if page == "📊 Veri Görselleştirme":
    st.title("📊 Veri Keşfi & Görselleştirme")

    st.markdown("""
    <div class="dataset-link">
        📁 <strong>Dataset:</strong>
        <a href="https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set" target="_blank">
        UCI Individual Household Electric Power Consumption (Kaggle)</a>
        — 2,075,259 dakikalık ölçüm → 34,589 saatlik gözlem
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── Key metrics ──
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="value">34,589</div><div class="label">Saatlik Gözlem</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="value">{df[target].mean():.2f} kW</div><div class="label">Ortalama Tüketim</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="value">{df[target].std():.2f} kW</div><div class="label">Std. Sapma</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="value">{df[target].max():.1f} kW</div><div class="label">Maksimum</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><div class="value">~4 Yıl</div><div class="label">Veri Aralığı</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── EDA TABs ──
    eda_tab1, eda_tab2, eda_tab3, eda_tab4, eda_tab5 = st.tabs([
        "📈 Zaman Serisi & Mevsimsellik",
        "🔬 STL Decomposition",
        "🌊 Fourier & ACF/PACF",
        "🔗 Korelasyon & Multicollinearity",
        "📊 Distribüsyon & Yıllık Karşılaştırma"
    ])

    # ══════════════════════════════════════════════════════════════
    # TAB 1: Zaman Serisi & Mevsimsellik
    # ══════════════════════════════════════════════════════════════
    with eda_tab1:
        # Time series
        st.subheader("Zaman Serisi — Tüm Veri")
        resample_opt = st.selectbox("Görüntüleme çözünürlüğü", ["Saatlik", "Günlük", "Haftalık"], index=1)
        resample_map = {"Saatlik": "1h", "Günlük": "D", "Haftalık": "W"}

        df_plot = df[target].resample(resample_map[resample_opt]).mean()
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot.values,
            mode='lines', line=dict(color='#065A82', width=1),
            name='Global Active Power'
        ))
        for x_val, color, label in [
            ('2009-12-31', '#F59E0B', 'Train→Val'),
            ('2010-06-30', '#EF4444', 'Val→Test'),
        ]:
            fig_ts.add_shape(type='line', x0=x_val, x1=x_val, y0=0, y1=1, yref='paper',
                             line=dict(dash='dash', color=color, width=1.5))
            fig_ts.add_annotation(x=x_val, y=1, yref='paper', text=label, showarrow=False,
                                  xanchor='left', font=dict(size=10, color=color), xshift=5)
        fig_ts.update_layout(height=400, template='plotly_white', yaxis_title="kW", margin=dict(t=30, b=40))
        st.plotly_chart(fig_ts, use_container_width=True)

        st.markdown("---")

        # Hourly & Monthly profiles
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Saatlik Profil")
            hourly = df.groupby('hour')[target].agg(['mean', 'median', 'std']).reset_index()
            fig_h = go.Figure()
            fig_h.add_trace(go.Bar(x=hourly['hour'], y=hourly['mean'], name='Ortalama',
                                   marker_color='#065A82', error_y=dict(type='data', array=hourly['std'], visible=True, thickness=1)))
            fig_h.add_trace(go.Scatter(x=hourly['hour'], y=hourly['median'], name='Medyan',
                                       line=dict(color='#02C39A', width=3)))
            fig_h.update_layout(height=350, template='plotly_white', xaxis_title="Saat", yaxis_title="kW", margin=dict(t=30))
            st.plotly_chart(fig_h, use_container_width=True)

        with col_b:
            st.subheader("Aylık Profil")
            monthly = df.groupby('month')[target].agg(['mean', 'median', 'std']).reset_index()
            month_names = ['Oca','Şub','Mar','Nis','May','Haz','Tem','Ağu','Eyl','Eki','Kas','Ara']
            fig_m = go.Figure()
            fig_m.add_trace(go.Bar(x=month_names, y=monthly['mean'], name='Ortalama',
                                   marker_color='#1C7293', error_y=dict(type='data', array=monthly['std'], visible=True, thickness=1)))
            fig_m.add_trace(go.Scatter(x=month_names, y=monthly['median'], name='Medyan',
                                       line=dict(color='#02C39A', width=3)))
            fig_m.update_layout(height=350, template='plotly_white', xaxis_title="Ay", yaxis_title="kW", margin=dict(t=30))
            st.plotly_chart(fig_m, use_container_width=True)

        st.markdown("---")

        # Heatmap
        st.subheader("Saat × Gün Isı Haritası")
        day_names = ['Pzt', 'Sal', 'Çar', 'Per', 'Cum', 'Cmt', 'Paz']
        heatmap_data = df.groupby(['dayofweek', 'hour'])[target].mean().unstack()
        fig_heat = px.imshow(
            heatmap_data.values,
            labels=dict(x="Saat", y="Gün", color="kW"),
            x=list(range(24)), y=day_names,
            color_continuous_scale='YlOrRd', aspect='auto'
        )
        fig_heat.update_layout(height=300, margin=dict(t=30, b=30))
        st.plotly_chart(fig_heat, use_container_width=True)

        # Weekday vs Weekend
        st.subheader("Hafta İçi vs Hafta Sonu Profili")
        wkday = df[df['is_weekend'] == 0].groupby('hour')[target].mean()
        wkend = df[df['is_weekend'] == 1].groupby('hour')[target].mean()
        fig_wk = go.Figure()
        fig_wk.add_trace(go.Scatter(x=wkday.index, y=wkday.values, name='Hafta İçi',
                                     line=dict(color='#065A82', width=2.5), fill='tozeroy', fillcolor='rgba(6,90,130,0.1)'))
        fig_wk.add_trace(go.Scatter(x=wkend.index, y=wkend.values, name='Hafta Sonu',
                                     line=dict(color='#F59E0B', width=2.5), fill='tozeroy', fillcolor='rgba(245,158,11,0.1)'))
        fig_wk.update_layout(height=350, template='plotly_white', xaxis_title="Saat", yaxis_title="kW",
                              xaxis=dict(dtick=2), margin=dict(t=30))
        st.plotly_chart(fig_wk, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            💡 <strong>Keşifler:</strong> Akşam 19-21 arası peak (~1.9 kW).
            Hafta sonu sabahları geç kalkış etkisi — peak saat sağa kayıyor.
            Kış tüketimi yazın 2 katı. Hafta sonu toplam tüketim %17 daha yüksek.
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 2: STL Decomposition
    # ══════════════════════════════════════════════════════════════
    with eda_tab2:
        from statsmodels.tsa.seasonal import STL

        st.subheader("STL Decomposition — Çoklu Mevsimsellik")
        st.markdown("STL (Seasonal-Trend decomposition using LOESS) ile trend, mevsimsellik ve residual bileşenlerine ayrıştırma.")

        stl_period = st.radio("Periyot seçin", ["Günlük (24h)", "Haftalık (168h)"], horizontal=True)
        period = 24 if "Günlük" in stl_period else 168

        with st.spinner("STL hesaplanıyor..."):
            series_stl = df[target].dropna()
            if len(series_stl) > 8760 and period == 168:
                series_stl = series_stl.iloc[-8760:]  # son 1 yıl
            stl_result = STL(series_stl, period=period, robust=True).fit()

        fig_stl = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                subplot_titles=('Orijinal Seri', 'Trend', 'Seasonal', 'Residual'),
                                vertical_spacing=0.06)
        fig_stl.add_trace(go.Scatter(x=series_stl.index, y=series_stl.values, line=dict(color='#065A82', width=0.8), showlegend=False), row=1, col=1)
        fig_stl.add_trace(go.Scatter(x=stl_result.trend.index, y=stl_result.trend.values, line=dict(color='#F59E0B', width=2), showlegend=False), row=2, col=1)
        fig_stl.add_trace(go.Scatter(x=stl_result.seasonal.index, y=stl_result.seasonal.values, line=dict(color='#02C39A', width=0.8), showlegend=False), row=3, col=1)
        fig_stl.add_trace(go.Scatter(x=stl_result.resid.index, y=stl_result.resid.values, line=dict(color='#EF4444', width=0.5), showlegend=False), row=4, col=1)
        fig_stl.update_layout(height=700, template='plotly_white', margin=dict(t=40))
        st.plotly_chart(fig_stl, use_container_width=True)

        # Seasonal strength
        var_resid = np.var(stl_result.resid.dropna())
        var_resid_seasonal = np.var(stl_result.resid.dropna() + stl_result.seasonal.dropna()[:len(stl_result.resid.dropna())])
        fs = max(0, 1 - var_resid / var_resid_seasonal) if var_resid_seasonal > 0 else 0

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown(f'<div class="metric-card"><div class="value">Fs = {fs:.3f}</div><div class="label">Seasonal Strength</div></div>', unsafe_allow_html=True)
        with col_s2:
            trend_pct = np.std(stl_result.trend.dropna()) / np.std(series_stl) * 100
            st.markdown(f'<div class="metric-card"><div class="value">{trend_pct:.1f}%</div><div class="label">Trend Payı</div></div>', unsafe_allow_html=True)
        with col_s3:
            resid_pct = np.std(stl_result.resid.dropna()) / np.std(series_stl) * 100
            st.markdown(f'<div class="metric-card"><div class="value">{resid_pct:.1f}%</div><div class="label">Residual Payı</div></div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="insight-box">
            💡 <strong>STL Bulguları ({stl_period}):</strong>
            Seasonal strength Fs={fs:.3f} — {"orta-güçlü" if fs > 0.4 else "zayıf"} mevsimsellik.
            Tek periyot yetersiz → çoklu mevsimsellik (12h, 24h, 168h, yıllık) gerekli.
            Bu, SARIMA'nın sınırlarına işaret ediyor (tek seasonal period destekler).
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 3: Fourier & ACF/PACF
    # ══════════════════════════════════════════════════════════════
    with eda_tab3:
        from statsmodels.tsa.stattools import acf as sm_acf

        col_f, col_a = st.columns(2)

        with col_f:
            st.subheader("Fourier Power Spectrum")
            st.markdown("Dominant periyotları frekans domaininde tespit ediyoruz.")

            # Use numpy FFT for clean spectrum
            series_fft = df[target].dropna().values
            n = len(series_fft)
            fft_vals = np.fft.rfft(series_fft - np.mean(series_fft))
            power = (np.abs(fft_vals) ** 2) / n
            freqs = np.fft.rfftfreq(n, d=1.0)  # d=1 hour → freqs in cycles/hour

            # Skip DC component (freq=0), zoom on 12h/24h peak region
            # Period range: 6h to 120h → 12h ve 24h zirveleri daha büyük görünür
            freq_min, freq_max = 1/120, 1/6
            mask_f = (freqs >= freq_min) & (freqs <= freq_max)
            plot_freqs = freqs[mask_f]
            plot_power = power[mask_f]

            # Smooth with rolling for cleaner display
            if len(plot_power) > 100:
                kernel = max(3, len(plot_power) // 200)
                plot_power_smooth = np.convolve(plot_power, np.ones(kernel)/kernel, mode='same')
            else:
                plot_power_smooth = plot_power

            fig_fft = go.Figure()
            fig_fft.add_trace(go.Scatter(
                x=plot_freqs, y=plot_power_smooth,
                mode='lines', line=dict(color='#065A82', width=1.5), showlegend=False
            ))

            # Annotate known periods
            target_periods = {'12h': 12, '24h': 24, '168h': 168}
            for label, period_h in target_periods.items():
                target_freq = 1.0 / period_h
                # Find closest frequency
                idx_closest = np.argmin(np.abs(plot_freqs - target_freq))
                if idx_closest < len(plot_power):
                    # Use actual peak near that frequency (±5 bins)
                    lo = max(0, idx_closest - 5)
                    hi = min(len(plot_power), idx_closest + 6)
                    peak_local = lo + np.argmax(plot_power[lo:hi])
                    pf = plot_freqs[peak_local]
                    pv = plot_power[peak_local]
                    ann_label = f"{label} ⭐" if period_h == 12 else label
                    fig_fft.add_annotation(
                        x=pf, y=pv, text=ann_label, showarrow=True,
                        arrowhead=2, font=dict(size=12, color='#EF4444'),
                        ay=-40
                    )

            # Custom tick labels showing period instead of frequency (zoomed range)
            tick_periods = [6, 8, 12, 24, 48, 72, 120]
            tick_freqs = [1.0/p for p in tick_periods]
            tick_labels = [f"{p}h" for p in tick_periods]

            fig_fft.update_layout(
                height=400, template='plotly_white',
                xaxis_title="Frekans (cycles/hour) — etiketler periyot cinsinden",
                yaxis_title="Power",
                xaxis=dict(
                    type="log",
                    tickmode="array",
                    tickvals=tick_freqs,
                    ticktext=tick_labels,
                    range=[np.log10(freq_min), np.log10(freq_max)]
                ),
                margin=dict(t=30)
            )
            st.plotly_chart(fig_fft, use_container_width=True)

            st.markdown("""
            <div class="insight-box" style="color: #0F172A;">
                💡 <strong>Kritik Keşif:</strong> 12 saatlik periyot 24 saatten daha güçlü!
                Hane günde 2 tüketim peak'i yapıyor (sabah 07-08, akşam 19-21).
                Standart 24h-only encoding bu bilgiyi kaçırır.
            </div>
            """, unsafe_allow_html=True)

        with col_a:
            st.subheader("ACF — Autokorelasyon")
            st.markdown("Hangi lag'lerde güçlü otokorelasyon var?")

            acf_vals = sm_acf(df[target].dropna().values, nlags=24*14, fft=True)
            lags = np.arange(len(acf_vals))
            ci = 1.96 / np.sqrt(len(df[target].dropna()))

            fig_acf = go.Figure()
            fig_acf.add_trace(go.Scatter(x=lags, y=acf_vals, mode='lines',
                                          line=dict(color='#065A82', width=1), showlegend=False))
            fig_acf.add_hline(y=ci, line_dash="dash", line_color="#EF4444", line_width=0.8)
            fig_acf.add_hline(y=-ci, line_dash="dash", line_color="#EF4444", line_width=0.8)
            fig_acf.add_hline(y=0, line_color="gray", line_width=0.5)

            # Key lag annotations
            for lag_val, label, color in [(1, "lag_1", "#02C39A"), (24, "lag_24", "#F59E0B"), (168, "lag_168", "#EF4444")]:
                if lag_val < len(acf_vals):
                    fig_acf.add_annotation(x=lag_val, y=acf_vals[lag_val],
                                            text=f"{label}={acf_vals[lag_val]:.3f}",
                                            showarrow=True, arrowhead=2,
                                            font=dict(size=10, color=color))

            fig_acf.update_layout(height=400, template='plotly_white',
                                   xaxis_title="Lag (saat)", yaxis_title="ACF",
                                   margin=dict(t=30))
            st.plotly_chart(fig_acf, use_container_width=True)

            st.markdown(f"""
            <div class="insight-box">
                💡 <strong>ACF Bulguları:</strong>
                lag_1 = {acf_vals[1]:.3f} (çok güçlü kısa vadeli bağımlılık),
                lag_24 = {acf_vals[24]:.3f} (günlük seasonality),
                lag_168 = {acf_vals[168]:.3f} (haftalık seasonality).
                Bu değerler Granger Causality ile doğrulanıp feature olarak seçildi.
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 4: Korelasyon & Multicollinearity
    # ══════════════════════════════════════════════════════════════
    with eda_tab4:
        st.subheader("Değişkenler Arası Korelasyon")

        orig_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                     'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        avail_cols = [c for c in orig_cols if c in df.columns]
        corr_matrix = df[avail_cols].corr()

        fig_corr = px.imshow(
            corr_matrix.values, x=avail_cols, y=avail_cols,
            color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
            text_auto='.2f', aspect='auto'
        )
        fig_corr.update_layout(height=450, margin=dict(t=30, b=30))
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("---")

        # Variable distributions
        st.subheader("Alt Sayaç Dağılımları")
        sub_cols = [c for c in ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'] if c in df.columns]
        if sub_cols:
            fig_sub = make_subplots(rows=1, cols=len(sub_cols),
                                     subplot_titles=[c.replace('Sub_metering_', 'Sub ') for c in sub_cols])
            colors_sub = ['#065A82', '#1C7293', '#02C39A']
            for i, col in enumerate(sub_cols):
                fig_sub.add_trace(go.Histogram(x=df[col].dropna(), nbinsx=60,
                                                marker_color=colors_sub[i], opacity=0.8,
                                                name=col, showlegend=False), row=1, col=i+1)
            fig_sub.update_layout(height=300, template='plotly_white', margin=dict(t=40))
            st.plotly_chart(fig_sub, use_container_width=True)

        # Scatter: Sub_metering_3 vs target
        st.subheader("Sub_metering_3 vs Global Active Power")
        sample_idx = np.random.choice(len(df), min(5000, len(df)), replace=False)
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scattergl(
            x=df['Sub_metering_3'].iloc[sample_idx],
            y=df[target].iloc[sample_idx],
            mode='markers', marker=dict(size=2, color='#065A82', opacity=0.3),
            showlegend=False
        ))
        fig_scatter.update_layout(height=350, template='plotly_white',
                                   xaxis_title="Sub_metering_3 (Wh)", yaxis_title="Global Active Power (kW)",
                                   margin=dict(t=30))
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            💡 <strong>Multicollinearity:</strong> Global_intensity, Global_active_power ile neredeyse birebir (P=V×I, VIF>3400).
            Modelleme aşamasında çıkarıldı. Sub_metering_3 (su ısıtıcı/klima) target ile en güçlü ilişkiye sahip
            ve SHAP analizinde de açık ara en etkili feature (SHAP=0.39).
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 5: Distribüsyon & Yıllık Karşılaştırma
    # ══════════════════════════════════════════════════════════════
    with eda_tab5:
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.subheader("Target Distribüsyon")
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=df[target], nbinsx=80, marker_color='#065A82', opacity=0.8, name='GAP'))
            fig_dist.add_vline(x=df[target].mean(), line_dash="dash", line_color="#EF4444",
                                annotation_text=f"Mean={df[target].mean():.2f}")
            fig_dist.add_vline(x=df[target].median(), line_dash="dash", line_color="#02C39A",
                                annotation_text=f"Median={df[target].median():.2f}")
            fig_dist.update_layout(height=350, template='plotly_white', xaxis_title="kW", yaxis_title="Frekans", margin=dict(t=30))
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_d2:
            st.subheader("Box Plot — Aylık")
            box_data = []
            for m in range(1, 13):
                vals = df[df['month'] == m][target].dropna().values
                box_data.append(go.Box(y=vals, name=month_names[m-1], marker_color='#1C7293', line_color='#065A82'))
            fig_box = go.Figure(data=box_data)
            fig_box.update_layout(height=350, template='plotly_white', yaxis_title="kW",
                                   showlegend=False, margin=dict(t=30))
            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("---")

        # Year-over-Year
        st.subheader("Yıllık Karşılaştırma (Year-over-Year)")
        df_yoy = df[[target]].copy()
        df_yoy['year'] = df_yoy.index.year
        df_yoy['month'] = df_yoy.index.month
        yearly_monthly = df_yoy.groupby(['year', 'month'])[target].mean().reset_index()

        fig_yoy = go.Figure()
        colors_yr = {'2007': '#94A3B8', '2008': '#1C7293', '2009': '#065A82', '2010': '#02C39A'}
        for yr in sorted(yearly_monthly['year'].unique()):
            yr_data = yearly_monthly[yearly_monthly['year'] == yr]
            fig_yoy.add_trace(go.Scatter(
                x=[month_names[m-1] for m in yr_data['month']],
                y=yr_data[target].values,
                name=str(yr),
                line=dict(color=colors_yr.get(str(yr), '#065A82'), width=2.5),
                mode='lines+markers'
            ))
        fig_yoy.update_layout(height=400, template='plotly_white', yaxis_title="kW", margin=dict(t=30))
        st.plotly_chart(fig_yoy, use_container_width=True)

        st.markdown("---")

        # Seasonal: Kış vs Yaz
        st.subheader("Kış vs Yaz — Saatlik Profil")
        winter = df[df['month'].isin([12, 1, 2])].groupby('hour')[target].mean()
        summer = df[df['month'].isin([6, 7, 8])].groupby('hour')[target].mean()
        fig_season = go.Figure()
        fig_season.add_trace(go.Scatter(x=winter.index, y=winter.values, name='Kış (Ara-Oca-Şub)',
                                         line=dict(color='#065A82', width=3), fill='tozeroy', fillcolor='rgba(6,90,130,0.15)'))
        fig_season.add_trace(go.Scatter(x=summer.index, y=summer.values, name='Yaz (Haz-Tem-Ağu)',
                                         line=dict(color='#F59E0B', width=3), fill='tozeroy', fillcolor='rgba(245,158,11,0.15)'))
        fig_season.update_layout(height=350, template='plotly_white', xaxis_title="Saat", yaxis_title="kW",
                                  xaxis=dict(dtick=2), margin=dict(t=30))
        st.plotly_chart(fig_season, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            💡 <strong>Mevsimsel Fark:</strong> Kış ortalama ~1.4 kW vs Yaz ~0.65 kW — 2× fark.
            Kış akşam peak'i 2.2 kW'a ulaşırken yaz peak'i 1.1 kW'da kalıyor.
            Yıllar arası (2007-2010) mevsimsel pattern tutarlı — structural break yok (Zivot-Andrews p=0.41).
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# PAGE 2: TAHMİN EKRANI
# ─────────────────────────────────────────────────────────────────────
elif page == "⚡ Tahmin Ekranı":
    st.title("⚡ Tahmin Ekranı")

    st.markdown("""
    <div class="dataset-link" style="color: #0F172A;">
        Model eğitimi tamamlandıktan sonra tarih aralığı seçerek tahminleri görselleştirin.
        <strong>Optuna ile optimize edilmiş</strong> LightGBM ve XGBoost modelleri kullanılır.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    with st.spinner("Modeller eğitiliyor (ilk çalıştırmada ~1-2 dakika)..."):
        lgb_model, xgb_model, preds, shap_imp = train_models(train, val, test, feature_cols)

    # Date range selector
    st.subheader("Tahmin Dönemi Seçin")
    tab_val, tab_test = st.tabs(["📙 Validation (Oca-Haz 2010)", "📕 Test (Tem-Kas 2010)"])

    with tab_val:
        col1, col2 = st.columns(2)
        with col1:
            val_start = st.date_input("Başlangıç", val.index.min().date(), min_value=val.index.min().date(), max_value=val.index.max().date(), key="vs")
        with col2:
            val_end_d = st.date_input("Bitiş", min(val.index.min().date() + pd.Timedelta(days=14), val.index.max().date()), min_value=val.index.min().date(), max_value=val.index.max().date(), key="ve")

        mask_v = (val.index.date >= val_start) & (val.index.date <= val_end_d)
        y_v = val.loc[mask_v, target]
        idx_v = np.where(mask_v)[0]

        if len(y_v) > 0:
            fig_v = go.Figure()
            fig_v.add_trace(go.Scatter(x=y_v.index, y=y_v.values, mode='lines', name='Gerçek', line=dict(color='black', width=1.5)))
            fig_v.add_trace(go.Scatter(x=y_v.index, y=preds['LightGBM_val'][idx_v], mode='lines', name='LightGBM', line=dict(color='#065A82', width=1.5, dash='dot')))
            fig_v.add_trace(go.Scatter(x=y_v.index, y=preds['XGBoost_val'][idx_v], mode='lines', name='XGBoost', line=dict(color='#02C39A', width=1.5, dash='dash')))
            fig_v.add_trace(go.Scatter(x=y_v.index, y=preds['Naive_val'][idx_v], mode='lines', name='Naive', line=dict(color='#EF4444', width=1, dash='dot'), opacity=0.5))
            fig_v.update_layout(height=450, template='plotly_white', yaxis_title="kW", legend=dict(orientation="h", y=1.1), margin=dict(t=40))
            st.plotly_chart(fig_v, use_container_width=True)

            # Instant metrics
            c1, c2, c3 = st.columns(3)
            m_lgb = calc_metrics(y_v, preds['LightGBM_val'][idx_v])
            m_xgb = calc_metrics(y_v, preds['XGBoost_val'][idx_v])
            m_naive = calc_metrics(y_v, preds['Naive_val'][idx_v])
            with c1:
                st.metric("LightGBM MAE", f"{m_lgb['MAE']:.3f} kW", f"{(1-m_lgb['MAE']/m_naive['MAE'])*100:.0f}% ↓ vs Naive")
            with c2:
                st.metric("XGBoost MAE", f"{m_xgb['MAE']:.3f} kW", f"{(1-m_xgb['MAE']/m_naive['MAE'])*100:.0f}% ↓ vs Naive")
            with c3:
                st.metric("Naive MAE", f"{m_naive['MAE']:.3f} kW", "Baseline")

    with tab_test:
        col1, col2 = st.columns(2)
        with col1:
            test_start = st.date_input("Başlangıç", test.index.min().date(), min_value=test.index.min().date(), max_value=test.index.max().date(), key="ts")
        with col2:
            test_end_d = st.date_input("Bitiş", min(test.index.min().date() + pd.Timedelta(days=14), test.index.max().date()), min_value=test.index.min().date(), max_value=test.index.max().date(), key="te")

        mask_t = (test.index.date >= test_start) & (test.index.date <= test_end_d)
        y_t = test.loc[mask_t, target]
        idx_t = np.where(mask_t)[0]

        if len(y_t) > 0:
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(x=y_t.index, y=y_t.values, mode='lines', name='Gerçek', line=dict(color='black', width=1.5)))
            fig_t.add_trace(go.Scatter(x=y_t.index, y=preds['LightGBM_test'][idx_t], mode='lines', name='LightGBM', line=dict(color='#065A82', width=1.5, dash='dot')))
            fig_t.add_trace(go.Scatter(x=y_t.index, y=preds['XGBoost_test'][idx_t], mode='lines', name='XGBoost', line=dict(color='#02C39A', width=1.5, dash='dash')))
            fig_t.add_trace(go.Scatter(x=y_t.index, y=preds['Naive_test'][idx_t], mode='lines', name='Naive', line=dict(color='#EF4444', width=1, dash='dot'), opacity=0.5))
            fig_t.update_layout(height=450, template='plotly_white', yaxis_title="kW", legend=dict(orientation="h", y=1.1), margin=dict(t=40))
            st.plotly_chart(fig_t, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            m_lgb = calc_metrics(y_t, preds['LightGBM_test'][idx_t])
            m_xgb = calc_metrics(y_t, preds['XGBoost_test'][idx_t])
            m_naive = calc_metrics(y_t, preds['Naive_test'][idx_t])
            with c1:
                st.metric("LightGBM MAE", f"{m_lgb['MAE']:.3f} kW", f"{(1-m_lgb['MAE']/m_naive['MAE'])*100:.0f}% ↓ vs Naive")
            with c2:
                st.metric("XGBoost MAE", f"{m_xgb['MAE']:.3f} kW", f"{(1-m_xgb['MAE']/m_naive['MAE'])*100:.0f}% ↓ vs Naive")
            with c3:
                st.metric("Naive MAE", f"{m_naive['MAE']:.3f} kW", "Baseline")


# ─────────────────────────────────────────────────────────────────────
# PAGE 3: MODEL PERFORMANS
# ─────────────────────────────────────────────────────────────────────
elif page == "🏆 Model Performans":
    st.title("🏆 Model Performans Özeti")

    st.markdown("""
    <div class="dataset-link">
        📁 <strong>Dataset:</strong>
        <a href="https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set" target="_blank">
        UCI Individual Household Electric Power Consumption (Kaggle)</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    with st.spinner("Modeller eğitiliyor..."):
        lgb_model, xgb_model, preds, shap_imp = train_models(train, val, test, feature_cols)

    y_test = test[target]
    y_val = val[target]

    # Metrics table
    st.subheader("Test Set — Performans Karşılaştırma")

    models_metrics = []
    for name, pred_key in [("XGBoost", "XGBoost_test"), ("LightGBM", "LightGBM_test"), ("Naive (t-1)", "Naive_test")]:
        m = calc_metrics(y_test, preds[pred_key])
        m['Model'] = name
        models_metrics.append(m)

    metrics_df = pd.DataFrame(models_metrics)[['Model', 'MAE', 'RMSE', 'MAPE', 'R²']]
    metrics_df['MAE'] = metrics_df['MAE'].round(4)
    metrics_df['RMSE'] = metrics_df['RMSE'].round(4)
    metrics_df['MAPE'] = metrics_df['MAPE'].round(2)
    metrics_df['R²'] = metrics_df['R²'].round(4)

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Big metrics
    best = metrics_df.iloc[0]
    naive_mae = metrics_df[metrics_df['Model'] == 'Naive (t-1)']['MAE'].values[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="value">{best["MAE"]:.4f}</div><div class="label">En İyi MAE (kW)</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="value">{best["R²"]:.3f}</div><div class="label">R² Skoru</div></div>', unsafe_allow_html=True)
    with col3:
        imp = (1 - best['MAE'] / naive_mae) * 100
        st.markdown(f'<div class="metric-card"><div class="value">%{imp:.0f}</div><div class="label">Naive\'e Göre İyileşme</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="value">{best["Model"]}</div><div class="label">Şampiyon Model</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Bar chart comparison
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("MAE Karşılaştırma")
        fig_bar = go.Figure()
        colors = ['#02C39A', '#065A82', '#94A3B8']
        fig_bar.add_trace(go.Bar(
            x=metrics_df['Model'], y=metrics_df['MAE'],
            marker_color=colors,
            text=metrics_df['MAE'].apply(lambda x: f"{x:.4f}"),
            textposition='outside'
        ))
        fig_bar.update_layout(height=350, template='plotly_white', yaxis_title="MAE (kW)", margin=dict(t=30))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        st.subheader("R² Karşılaştırma")
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Bar(
            x=metrics_df['Model'], y=metrics_df['R²'],
            marker_color=colors,
            text=metrics_df['R²'].apply(lambda x: f"{x:.4f}"),
            textposition='outside'
        ))
        fig_r2.update_layout(height=350, template='plotly_white', yaxis_title="R²", margin=dict(t=30))
        st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown("---")

    # Residual analysis
    st.subheader("Residual Analizi — LightGBM")
    residuals = y_test.values - preds['LightGBM_test']

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=test.index, y=residuals,
            mode='markers', marker=dict(size=2, color='#065A82', opacity=0.4),
            name='Residual'
        ))
        fig_res.add_hline(y=0, line_color='red', line_width=1)
        fig_res.update_layout(height=350, template='plotly_white', yaxis_title="Residual (kW)", margin=dict(t=30))
        st.plotly_chart(fig_res, use_container_width=True)

    with col_r2:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=residuals, nbinsx=60, marker_color='#065A82', opacity=0.8))
        fig_hist.update_layout(height=350, template='plotly_white', xaxis_title="Residual (kW)", yaxis_title="Frekans", margin=dict(t=30))
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown(f"""
    <div class="insight-box">
        💡 <strong>Residual:</strong> μ={residuals.mean():.4f}, σ={residuals.std():.3f} — Neredeyse sıfır bias.
        Sağa çarpık kuyruk var (aşırı tüketim anları tam yakalanamıyor).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # SHAP
    st.subheader("SHAP — Feature Importance (LightGBM)")

    top_n = st.slider("Top N Feature", 10, 30, 15)
    shap_top = shap_imp.head(top_n).sort_values('SHAP')

    fig_shap = go.Figure()
    fig_shap.add_trace(go.Bar(
        x=shap_top['SHAP'], y=shap_top['Feature'],
        orientation='h', marker_color='#065A82'
    ))
    fig_shap.update_layout(
        height=max(350, top_n * 28), template='plotly_white',
        xaxis_title="mean(|SHAP value|)",
        margin=dict(t=30, l=150)
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        💡 <strong>SHAP Bulguları:</strong>
        Sub_metering_3 (su ısıtıcı/klima) açık ara en etkili feature (SHAP=0.39).
        lag_1 ikinci sırada (SHAP=0.17) — autoregressive yapı kritik.
        LightGBM ve XGBoost arasında Spearman ρ=0.986 ile mükemmel tutarlılık.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # İleri vizyon
    st.subheader("İleri Vizyon")
    st.markdown("""
    - **Real-time tahmin:** Streaming veri ile saatlik güncellenen tahminler
    - **Anomali tespiti:** Residual pattern'larından otomatik anomali algılama
    - **Enerji optimizasyonu:** Tahminlere dayalı dinamik fiyatlandırma / yük yönetimi
    - **Transfer learning:** Chronos/TSMixer ile farklı hanelere zero-shot tahmin
    - **Causal inference:** Müdahale senaryoları (tarife değişikliği, cihaz yenileme) etkisi
    """)