from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from screener.config import APP_TITLE, DB_PATH
from screener.heuristics import mandate_fit_row, rating_value
from screener.metrics import build_price_matrix, correlation_matrix, hedge_ratios, portfolio_metrics, portfolio_series, weighted_portfolio_attributes
from screener.repository import get_app_meta, get_mandates, get_price_history, get_refresh_log, get_universe
from screener.seed import ensure_seeded, reset_and_seed
from screener.service import ensure_history_for_selection, refresh_all, refresh_etf_universe, refresh_public_try, refresh_tefas_universe

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")

@st.cache_data(show_spinner=False)
def load_data(cache_buster: int = 0):
    ensure_seeded(DB_PATH)
    return get_universe(DB_PATH), get_mandates(DB_PATH), get_refresh_log(DB_PATH), get_app_meta(DB_PATH)

@st.cache_data(show_spinner=False)
def load_history(ids: tuple[str, ...], cache_buster: int = 0):
    return get_price_history(DB_PATH, list(ids))


def fmt_pct(x):
    return '-' if pd.isna(x) else f'%{x:,.2f}'


def fmt_dt(x):
    ts = pd.to_datetime(x, errors='coerce')
    return '-' if pd.isna(ts) else ts.strftime('%d.%m.%Y %H:%M')

if 'cache_buster' not in st.session_state:
    st.session_state.cache_buster = 0

universe, mandates, refresh_log, meta = load_data(st.session_state.cache_buster)
meta_map = dict(zip(meta['key'], meta['value'])) if not meta.empty else {}

st.title('📊 Fund Manager Workstation v3')
st.caption("Tek web uygulamasında binlerce enstrüman evreni, mandate fit, compare ve portfolio lab. İlk tuş universe'ü kurar; seçilen ETF ve TEFAS fonlarının tarihçesi gerektiğinde otomatik çekilir.")

with st.sidebar:
    st.subheader('Kontrol Merkezi')
    if st.button('Veritabanını sıfırla ve temel yapıyı yeniden kur', use_container_width=True):
        result = reset_and_seed(DB_PATH)
        st.session_state.cache_buster += 1
        st.success(f'Kuruldu: {result.instruments} bootstrap enstrüman, {result.mandates} mandate, {result.prices} demo tarihçe satırı.')
        st.rerun()
    if st.button('Tüm evreni yenile (TEFAS + Nasdaq ETF + TRY referans)', type='primary', use_container_width=True):
        summary = refresh_all(DB_PATH)
        st.session_state.cache_buster += 1
        for run in summary.runs:
            st.write(('✅' if run.status == 'ok' else '⚠️') + f' {run.source}: {run.message}')
        st.rerun()
    c1, c2, c3 = st.columns(3)
    if c1.button('TEFAS', use_container_width=True):
        run = refresh_tefas_universe(DB_PATH)
        st.session_state.cache_buster += 1
        st.success(run.message)
        st.rerun()
    if c2.button('ETF', use_container_width=True):
        run = refresh_etf_universe(DB_PATH)
        st.session_state.cache_buster += 1
        st.success(run.message)
        st.rerun()
    if c3.button('TRY', use_container_width=True):
        run = refresh_public_try(DB_PATH)
        st.session_state.cache_buster += 1
        st.success(run.message)
        st.rerun()
    st.markdown('---')
    if not mandates.empty:
        mandate_id = st.selectbox('Aktif mandate', mandates['mandate_id'].tolist(), format_func=lambda x: mandates.loc[mandates['mandate_id']==x,'name'].iloc[0])
        selected_mandate = mandates[mandates['mandate_id'] == mandate_id].iloc[0]
        st.info(
            f"Baz döviz: {selected_mandate['base_currency']}\n\n"
            f"İzinli sınıflar: {selected_mandate['allowed_asset_classes']}\n\n"
            f"Min rating: {selected_mandate['min_rating']}"
        )
    else:
        selected_mandate = pd.Series(dtype='object')
    st.caption(f"Son seed: {fmt_dt(meta_map.get('last_seeded_at'))}")
    st.caption(f"Son TEFAS refresh: {fmt_dt(meta_map.get('last_refresh_tefas'))}")
    st.caption(f"Son ETF universe refresh: {fmt_dt(meta_map.get('last_refresh_etf_universe'))}")
    st.caption(f"Son TRY ref refresh: {fmt_dt(meta_map.get('last_refresh_public_try'))}")

if not universe.empty and not selected_mandate.empty:
    fit = universe.apply(lambda r: mandate_fit_row(r, selected_mandate), axis=1, result_type='expand')
    universe = universe.copy()
    universe['mandate_fit'] = fit[0]
    universe['mandate_note'] = fit[1]
else:
    universe['mandate_fit'] = 'Pass'
    universe['mandate_note'] = 'Mandate yok'
universe['data_status'] = universe['is_demo'].map({0:'Canlı/Kamu',1:'Seed'}).fillna('Canlı/Kamu')

market_tab, screener_tab, compare_tab, portfolio_tab, mandate_tab, data_tab = st.tabs(['Market Map','Screener','Compare','Portfolio Lab','Mandates','Data Console'])

with market_tab:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Toplam enstrüman', f'{len(universe):,}'.replace(',', '.'))
    c2.metric('Asset class', f"{universe['asset_class'].nunique()}")
    c3.metric('Canlı/Kamu', f"{(universe['is_demo'] == 0).sum():,}".replace(',', '.'))
    c4.metric('Pass', f"{(universe['mandate_fit'] == 'Pass').sum():,}".replace(',', '.'))
    left, right = st.columns([1.3,1])
    with left:
        if not universe.empty:
            counts = universe.groupby(['asset_class','data_status']).size().reset_index(name='count')
            fig = px.bar(counts, x='asset_class', y='count', color='data_status', barmode='group', title='Asset class dağılımı')
            st.plotly_chart(fig, use_container_width=True)
    with right:
        top = universe.sort_values(['aum_mn','ret_1y_pct','liquidity_score'], ascending=[False,False,False]).head(15)
        st.dataframe(top[['name','asset_class','category','currency','ret_1y_pct','aum_mn','liquidity_score','data_status','mandate_fit']].rename(columns={'name':'Enstrüman','asset_class':'Asset','category':'Kategori','currency':'Döviz','ret_1y_pct':'1Y Getiri','aum_mn':'AUM mn','liquidity_score':'Likidite','data_status':'Veri','mandate_fit':'Fit'}), use_container_width=True, hide_index=True)

with screener_tab:
    st.markdown('### Universal Screener')
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        asset_filter = st.multiselect('Asset class', sorted(universe['asset_class'].dropna().unique()), default=[])
        category_filter = st.multiselect('Kategori', sorted(universe['category'].dropna().astype(str).unique()), default=[])
    with f2:
        currency_filter = st.multiselect('Döviz', sorted(universe['currency'].dropna().astype(str).unique()), default=[])
        theme_filter = st.multiselect('Tema', sorted([x for x in universe['theme'].dropna().astype(str).unique() if x]), default=[])
    with f3:
        min_esg = st.slider('Minimum ESG', 0, 100, 0)
        min_liq = st.slider('Minimum likidite', 0, 100, 0)
    with f4:
        max_duration = st.slider('Maksimum duration (yıl)', 0.0, 15.0, 15.0, 0.5)
        min_ret = st.number_input('Minimum 1Y getiri', value=-100.0, step=5.0)
    s1, s2, s3, s4 = st.columns([1.5,1,1,1])
    with s1:
        search = st.text_input('Ara', placeholder='green esg, treasury, eurobond, ppf...')
    with s2:
        only_pass = st.checkbox('Sadece Pass', value=False)
    with s3:
        only_live = st.checkbox('Sadece canlı/kamu', value=False)
    with s4:
        sort_by = st.selectbox('Sırala', ['Mandate fit','1Y getiri yüksekten','Likidite yüksekten','ESG yüksekten','AUM yüksekten','Duration kısadan'])
    filt = universe.copy()
    if asset_filter:
        filt = filt[filt['asset_class'].isin(asset_filter)]
    if category_filter:
        filt = filt[filt['category'].isin(category_filter)]
    if currency_filter:
        filt = filt[filt['currency'].isin(currency_filter)]
    if theme_filter:
        filt = filt[filt['theme'].isin(theme_filter)]
    filt = filt[(filt['esg_score'].fillna(0) >= min_esg) & (filt['liquidity_score'].fillna(0) >= min_liq)]
    filt = filt[(filt['duration_years'].fillna(0) <= max_duration) & (filt['ret_1y_pct'].fillna(-999) >= min_ret)]
    if search:
        q = search.lower().strip()
        mask = filt['name'].astype(str).str.lower().str.contains(q) | filt['category'].astype(str).str.lower().str.contains(q) | filt['theme'].astype(str).str.lower().str.contains(q) | filt['asset_class'].astype(str).str.lower().str.contains(q)
        filt = filt[mask]
    if only_pass:
        filt = filt[filt['mandate_fit'] == 'Pass']
    if only_live:
        filt = filt[filt['is_demo'] == 0]
    if sort_by == 'Mandate fit':
        filt = filt.assign(_fit=filt['mandate_fit'].map({'Pass':0,'Warning':1,'Block':2}).fillna(3), _rating=filt['rating'].map(rating_value)).sort_values(['_fit','_rating','liquidity_score','ret_1y_pct'], ascending=[True,False,False,False]).drop(columns=['_fit','_rating'])
    elif sort_by == '1Y getiri yüksekten':
        filt = filt.sort_values('ret_1y_pct', ascending=False)
    elif sort_by == 'Likidite yüksekten':
        filt = filt.sort_values('liquidity_score', ascending=False)
    elif sort_by == 'ESG yüksekten':
        filt = filt.sort_values('esg_score', ascending=False)
    elif sort_by == 'AUM yüksekten':
        filt = filt.sort_values('aum_mn', ascending=False)
    else:
        filt = filt.sort_values('duration_years', ascending=True)
    st.write(f"Sonuç: {len(filt):,} enstrüman".replace(',', '.'))
    disp = filt[['instrument_id','name','asset_class','sub_asset_class','category','currency','rating','duration_years','yield_pct','ytm_pct','ret_1y_pct','vol_1y_pct','max_drawdown_1y_pct','esg_score','liquidity_score','aum_mn','theme','data_status','mandate_fit','mandate_note']].rename(columns={'instrument_id':'Kod','name':'Enstrüman','asset_class':'Asset','sub_asset_class':'Alt Sınıf','category':'Kategori','currency':'Döviz','rating':'Rating','duration_years':'Duration','yield_pct':'Getiri','ytm_pct':'YTM','ret_1y_pct':'1Y Getiri','vol_1y_pct':'Vol','max_drawdown_1y_pct':'Max DD','esg_score':'ESG','liquidity_score':'Likidite','aum_mn':'AUM mn','theme':'Tema','data_status':'Veri','mandate_fit':'Fit','mandate_note':'Not'})
    st.dataframe(disp, use_container_width=True, hide_index=True)

with compare_tab:
    st.markdown('### Compare Desk')
    default_ids = universe['instrument_id'].head(3).tolist() if not universe.empty else []
    compare_ids = st.multiselect('Karşılaştırılacak enstrümanlar', universe['instrument_id'].tolist(), default=default_ids, format_func=lambda x: universe.loc[universe['instrument_id']==x,'name'].iloc[0])
    if compare_ids:
        runs = ensure_history_for_selection(DB_PATH, compare_ids)
        if runs.runs:
            st.info('Eksik tarihçeler otomatik çekildi.')
            st.session_state.cache_buster += 1
        comp = universe[universe['instrument_id'].isin(compare_ids)].copy()
        st.dataframe(comp[['name','asset_class','category','currency','rating','duration_years','yield_pct','ytm_pct','ret_1y_pct','vol_1y_pct','max_drawdown_1y_pct','esg_score','liquidity_score','aum_mn','data_status','mandate_fit','mandate_note']].rename(columns={'name':'Enstrüman','asset_class':'Asset','category':'Kategori','currency':'Döviz','rating':'Rating','duration_years':'Duration','yield_pct':'Getiri','ytm_pct':'YTM','ret_1y_pct':'1Y Getiri','vol_1y_pct':'Vol','max_drawdown_1y_pct':'Max DD','esg_score':'ESG','liquidity_score':'Likidite','aum_mn':'AUM mn','data_status':'Veri','mandate_fit':'Fit','mandate_note':'Not'}), use_container_width=True, hide_index=True)
        history = load_history(tuple(compare_ids), st.session_state.cache_buster)
        prices = build_price_matrix(history)
        if not prices.empty:
            norm = prices / prices.ffill().iloc[0] * 100
            fig = px.line(norm.reset_index(), x='date', y=norm.columns, title='Normalize fiyat performansı')
            st.plotly_chart(fig, use_container_width=True)
            corr = correlation_matrix(prices)
            if not corr.empty:
                st.dataframe(corr, use_container_width=True)
            hedge = hedge_ratios(prices)
            if not hedge.empty:
                st.dataframe(hedge.sort_values('Beta/Hedge Ratio', key=lambda s: s.abs()), use_container_width=True, hide_index=True)
        else:
            st.warning('Bu seçim için tarihçe bulunamadı.')

with portfolio_tab:
    st.markdown('### Portfolio Lab')
    portfolio_ids = st.multiselect('Sepete eklenecek enstrümanlar', universe['instrument_id'].tolist(), default=default_ids, format_func=lambda x: universe.loc[universe['instrument_id']==x,'name'].iloc[0], key='portfolio_ids')
    weights = {}
    if portfolio_ids:
        ensure_history_for_selection(DB_PATH, portfolio_ids)
        st.session_state.cache_buster += 1
        cols = st.columns(min(4, len(portfolio_ids)))
        for i, iid in enumerate(portfolio_ids):
            name = universe.loc[universe['instrument_id']==iid,'name'].iloc[0]
            weights[iid] = cols[i % len(cols)].number_input(name[:28], min_value=0.0, max_value=100.0, value=float(round(100/len(portfolio_ids),2)), step=1.0)
        hist = load_history(tuple(portfolio_ids), st.session_state.cache_buster)
        prices = build_price_matrix(hist)
        series = portfolio_series(weights, prices)
        if not series.empty:
            fig = px.line(series.reset_index(), x='date', y=0, title='Portföy endeksi')
            st.plotly_chart(fig, use_container_width=True)
            mets = portfolio_metrics(series)
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            for col, (k, v) in zip([c1,c2,c3,c4,c5,c6], mets.items()):
                col.metric(k, fmt_pct(v) if '%' in k else f"{v:,.2f}")
            attrs = weighted_portfolio_attributes(universe, weights)
            if not attrs.empty:
                st.dataframe(attrs, use_container_width=True, hide_index=True)
            corr = correlation_matrix(prices)
            if not corr.empty:
                st.markdown('#### Korelasyon matrisi')
                st.dataframe(corr, use_container_width=True)
        else:
            st.warning('Portföy için tarihçe bulunamadı.')

with mandate_tab:
    st.markdown('### Mandates')
    st.dataframe(mandates, use_container_width=True, hide_index=True)
    st.markdown('#### Aktif mandate altında ilk 20 Pass')
    st.dataframe(universe[universe['mandate_fit']=='Pass'][['name','asset_class','category','currency','rating','duration_years','ret_1y_pct','liquidity_score','esg_score']].head(20).rename(columns={'name':'Enstrüman','asset_class':'Asset','category':'Kategori','currency':'Döviz','rating':'Rating','duration_years':'Duration','ret_1y_pct':'1Y Getiri','liquidity_score':'Likidite','esg_score':'ESG'}), use_container_width=True, hide_index=True)

with data_tab:
    st.markdown('### Data Console')
    st.write('Bu bölümde seed ile gelen bootstrap seti ile sonradan çekilen kamu/canlı evreni aynı yerde izlersin.')
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('#### Son refresh logları')
        st.dataframe(refresh_log, use_container_width=True, hide_index=True)
    with c2:
        dist = universe.groupby(['source_type']).size().reset_index(name='count').sort_values('count', ascending=False)
        st.dataframe(dist, use_container_width=True, hide_index=True)
    st.markdown('#### Evren önizleme')
    st.dataframe(universe.head(200), use_container_width=True, hide_index=True)
