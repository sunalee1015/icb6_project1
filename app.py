import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# 페이지 설정
st.set_page_config(page_title="셀러 운영 효율성 대시보드", layout="wide")

# ------------------------------------------------------------------------------
# 1. 데이터 로드 및 분석 엔진 (Core Logic)
# ------------------------------------------------------------------------------
@st.cache_data
def load_and_analyze_data(file_path):
    df = pd.read_excel(file_path)
    df['주문일'] = pd.to_datetime(df['주문일'])
    
    # [요구사항 1] 컬럼 정의 정정 및 리네이밍
    # 기존 '공급단가' -> 'supply_total_cost(공급가총합)'
    df = df.rename(columns={'공급단가': 'supply_total_cost'})
    
    # [요구사항 2] 수익성 지표 재계산 (분모 0 처리 추가)
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # 1) 마진율 (Gross Margin Rate)
    df['margin_ratio'] = np.where(
        df['실결제 금액'] != 0, 
        (df['실결제 금액'] - df['supply_total_cost']) / df['실결제 금액'], 
        np.nan
    )
    # 2) 원가대비 마진율 (Markup Rate)
    df['markup_ratio'] = np.where(
        df['supply_total_cost'] != 0, 
        (df['실결제 금액'] - df['supply_total_cost']) / df['supply_total_cost'], 
        np.nan
    )
    # [요구사항] 프리미엄 신규 정의 반영 (OR 조건)
    # 1. 상품 유형이 '선물세트'일 것
    cond_gift = (df['선물세트_여부'] == '선물세트')
    # 2. 감귤류 로얄과
    cond_citrus_royal = (df['품종'] == '감귤') & (df['과수 크기'] == '로얄과')
    # 3. 만감류 중과 이상 (중과, 중대과, 대과 포함)
    cond_mangam_large = (df['품종'].isin(['황금향', '한라봉', '레드향', '천혜향'])) & \
                        (df['과수 크기'].isin(['중과', '중대과, 대과', '대과']))
    
    df['is_premium'] = (cond_gift | cond_citrus_royal | cond_mangam_large).astype(int)
    
    df['is_cancel'] = df['취소여부'].apply(lambda x: 1 if x == 'Y' else 0)
    df['is_event'] = df['이벤트 여부'].apply(lambda x: 1 if x == 'Y' else 0)
    df['is_single'] = df['주문수량'].apply(lambda x: 1 if x == 1 else 0)

    # [재구매율 정의 반영] 24시간 이후 모든 상품 구매
    df_sorted = df.sort_values(by=['주문자연락처', '주문일'])
    df_sorted['prev_order_time'] = df_sorted.groupby('주문자연락처')['주문일'].shift(1)
    df_sorted['time_diff_hours'] = (df_sorted['주문일'] - df_sorted['prev_order_time']).dt.total_seconds() / 3600
    df_sorted['is_repurchase'] = df_sorted['time_diff_hours'].apply(lambda x: 1 if x >= 24 else 0)
    
    # [지역 클러스터링 로직]
    # '광역지역(정식)' 컬럼을 분석 기준으로 사용
    df['region_sido'] = df['광역지역(정식)']
    
    region_metrics = df.groupby('region_sido').agg({
        '실결제 금액': 'mean',
        '주문수량': 'mean',
        'is_premium': 'mean',
        'is_event': 'mean',
        'is_single': 'mean'
    }).rename(columns={
        '실결제 금액': 'avg_revenue',
        '주문수량': 'avg_qty',
        'is_premium': 'premium_ratio',
        'is_event': 'event_ratio',
        'is_single': 'single_ratio'
    })
    
    # K-Means 클러스터링 (설명용 보조 지표)
    X = region_metrics.fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    region_metrics['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 클러스터 특성에 따른 이름 매핑 (데이터 기반 정성적 해석)
    # [정정 요건] 지리적 명칭 배제 및 구매 행동 중심 네이밍
    cluster_stats = region_metrics.groupby('cluster').mean()
    
    # 매핑 로직 (프리미엄 비중 기준 정렬 순서 유지)
    rank = cluster_stats['premium_ratio'].sort_values(ascending=False).index
    cluster_map = {
        rank[0]: '소량·프리미엄형', 
        rank[1]: '이벤트·혼합형', 
        rank[2]: '대용량·가성비형'
    }
    region_metrics['region_type'] = region_metrics['cluster'].map(cluster_map)
    
    # 원본 데이터에 지역 타입 매핑
    df = df.merge(region_metrics[['region_type']], on='region_sido', how='left')
    
    # 셀러 단위 집계
    seller_data = df_sorted.groupby('셀러명').agg({
        '상품성등급_그룹': lambda x: (x == '프리미엄').mean(),
        '이벤트 여부': lambda x: (x == 'Y').mean(),
        '주문번호': 'count',
        '판매단가': 'mean',
        '실결제 금액': 'sum',
        'margin_ratio': 'mean',
        'markup_ratio': 'mean',
        'is_cancel': 'mean',
        'is_repurchase': 'sum'
    }).rename(columns={
        '상품성등급_그룹': 'premium_ratio',
        '이벤트 여부': 'event_ratio',
        '주문번호': 'order_count',
        '판매단가': 'avg_price',
        '실결제 금액': 'total_revenue',
        'margin_ratio': 'avg_margin',
        'markup_ratio': 'avg_markup',
        'is_cancel': 'cancel_rate',
        'is_repurchase': 'repurchase_count'
    })
    
    seller_data['repurchase_rate'] = seller_data['repurchase_count'] / seller_data['order_count']
    seller_data['aov'] = seller_data['total_revenue'] / seller_data['order_count']

    # [셀러 운영 유형 정량적 정의서 반영]
    def classify_seller(row):
        if row['premium_ratio'] >= 0.3: return '프리미엄 집중형'
        elif row['event_ratio'] >= 0.3: return '이벤트 의존형'
        elif row['order_count'] >= 15 and row['avg_price'] <= 28000: return '박리다매 운영형'
        else: return '일반 운영형'

    seller_data['seller_type'] = seller_data.apply(classify_seller, axis=1)
    df = df.merge(seller_data[['seller_type']], on='셀러명', how='left')
    
    return df, seller_data, region_metrics

# 데이터 로드
try:
    FILE_PATH = "preprocessed_data_2026.xlsx"
    df_raw, seller_metrics, region_stats = load_and_analyze_data(FILE_PATH)
except Exception as e:
    st.error(f"데이터를 로드하는 중 오류가 발생했습니다: {e}")
    st.stop()

# ------------------------------------------------------------------------------
# 2. 사이드바 필터
# ------------------------------------------------------------------------------
st.sidebar.title("📊 분석 원칙 및 필터")
st.sidebar.info("EDA 단계에서 정의된 정량적 기준을 바탕으로 분석을 시각화합니다.")

# 기간 필터
min_date = df_raw['주문일'].min().date()
max_date = df_raw['주문일'].max().date()
selected_date_range = st.sidebar.date_input("분석 기간 선택", [min_date, max_date], min_value=min_date, max_value=max_date)

# 셀러 유형 필터
seller_types = ['전체'] + list(seller_metrics['seller_type'].unique())
selected_type = st.sidebar.selectbox("셀러 유형 선택", seller_types)

# 데이터 필터링 적용
start_date, end_date = selected_date_range
mask = (df_raw['주문일'].dt.date >= start_date) & (df_raw['주문일'].dt.date <= end_date)
if selected_type != '전체':
    mask &= (df_raw['seller_type'] == selected_type)
filtered_df = df_raw[mask]

# [수정포인트 1] KPI 비교를 위한 직전 기간 데이터 필터링
period_delta = (end_date - start_date).days + 1
prev_start = start_date - timedelta(days=period_delta)
prev_end = start_date - timedelta(days=1)
prev_mask = (df_raw['주문일'].dt.date >= prev_start) & (df_raw['주문일'].dt.date <= prev_end)
if selected_type != '전체':
    prev_mask &= (df_raw['seller_type'] == selected_type)
prev_df = df_raw[prev_mask]

# 필터링된 셀러 지표 재계산 (상태 동기화)
curr_seller_metrics = seller_metrics.copy()
if selected_type != '전체':
    curr_seller_metrics = curr_seller_metrics[curr_seller_metrics['seller_type'] == selected_type]

# ------------------------------------------------------------------------------
# 3. 메인 레이아웃 및 탭 구성
# ------------------------------------------------------------------------------
st.title("🍊 셀러 운영 효율성 종합 대시보드")
st.markdown("---")

# 가~마 영역을 위한 탭 생성
tabs = st.tabs(["가. 종합 개요", "나. 셀러 유형 분석", "라. 상품 구조 분석", "마. 지역 소비 패턴 요약", "다. 상세 데이터 탐색"])

# ------------------------------------------------------------------------------
# 탭 가. 종합 개요 (Overview)
# ------------------------------------------------------------------------------
with tabs[0]:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    def get_delta(curr_val, prev_val, is_percent=False):
        if prev_val == 0 or pd.isna(prev_val): return None
        if is_percent: return f"{(curr_val - prev_val):.1%}"
        return f"{(curr_val - prev_val):,.0f}"

    with col1:
        curr_rev = filtered_df['실결제 금액'].sum()
        prev_rev = prev_df['실결제 금액'].sum()
        st.metric("총 매출액 (GMV)", f"{curr_rev:,.0f}원", delta=get_delta(curr_rev, prev_rev))
    with col2:
        curr_margin = filtered_df['margin_ratio'].mean()
        prev_margin = prev_df['margin_ratio'].mean()
        st.metric("평균 마진율", f"{curr_margin:.1%}", delta=get_delta(curr_margin, prev_margin, True))
    with col3:
        curr_eff = 1 - filtered_df['is_cancel'].mean()
        prev_eff = 1 - prev_df['is_cancel'].mean()
        st.metric("주문 완료 효율", f"{curr_eff:.1%}", delta=get_delta(curr_eff, prev_eff, True))
    with col4:
        premium_share = (filtered_df[filtered_df['상품성등급_그룹'] == '프리미엄']['실결제 금액'].sum() / filtered_df['실결제 금액'].sum()) if not filtered_df.empty else 0
        prev_premium_share = (prev_df[prev_df['상품성등급_그룹'] == '프리미엄']['실결제 금액'].sum() / prev_df['실결제 금액'].sum()) if not prev_df.empty else 0
        st.metric("프리미엄 매출 비중", f"{premium_share:.1%}", delta=get_delta(premium_share, prev_premium_share, True))
    with col5:
        event_share = (filtered_df[filtered_df['이벤트 여부'] == 'Y']['실결제 금액'].sum() / filtered_df['실결제 금액'].sum()) if not filtered_df.empty else 0
        prev_event_share = (prev_df[prev_df['이벤트 여부'] == 'Y']['실결제 금액'].sum() / prev_df['실결제 금액'].sum()) if not prev_df.empty else 0
        st.metric("이벤트 매출 비중", f"{event_share:.1%}", delta=get_delta(event_share, prev_event_share, True))
    
    st.caption(f"※ 하단 Delta(▲▼)는 직전 동일 기간({prev_start} ~ {prev_end}) 대비 변화량입니다.")
    st.markdown("###")
    # 대시보드 주요 현황 시각화 요약
    daily_rev_total = filtered_df.groupby(filtered_df['주문일'].dt.date)['실결제 금액'].sum().reset_index()
    daily_rev_total.columns = ['Date', 'Revenue']
    fig_overview = px.line(daily_rev_total, x='Date', y='Revenue', title="전체 매출 추이")
    st.plotly_chart(fig_overview, use_container_width=True)

# ------------------------------------------------------------------------------
# 탭 나. 셀러 유형 분석
# ------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("👨‍💼 셀러 운영 전략 유형별 성과")
    t_col1, t_col2 = st.columns(2)

    # 유형별 요약 데이터 생성
    type_summary = curr_seller_metrics.groupby('seller_type').agg({
        'total_revenue': 'sum',
        'avg_margin': 'mean',
        'repurchase_rate': 'mean',
        'aov': 'mean'
    }).reset_index()

    # [수정포인트] 전체 평균(Benchmark) 산출
    overall_avg_margin = curr_seller_metrics['avg_margin'].mean()
    overall_avg_repurchase = curr_seller_metrics['repurchase_rate'].mean()
    overall_avg_aov = curr_seller_metrics['aov'].mean()
    # 매출은 합계이므로 의미상 '유형당 평균 매출'로 기준 설정
    overall_avg_type_revenue = type_summary['total_revenue'].mean()

    with t_col1:
        fig_rev = px.bar(type_summary, x='seller_type', y='total_revenue', title="유형별 매출 기여도", color='seller_type', text_auto='.3s')
        fig_rev.add_hline(y=overall_avg_type_revenue, line_dash="dash", line_color="red", annotation_text="유형평균")
        st.plotly_chart(fig_rev, use_container_width=True)

        fig_rep = px.bar(type_summary, x='seller_type', y='repurchase_rate', title="유형별 평균 재구매율 (Retention)", color='seller_type', text_auto='.1%')
        fig_rep.add_hline(y=overall_avg_repurchase, line_dash="dash", line_color="red", annotation_text=f"전체평균({overall_avg_repurchase:.1%})")
        st.plotly_chart(fig_rep, use_container_width=True)

    with t_col2:
        fig_margin = px.bar(type_summary, x='seller_type', y='avg_margin', title="유형별 평균 마진율", color='seller_type', text_auto='.1%')
        fig_margin.add_hline(y=overall_avg_margin, line_dash="dash", line_color="red", annotation_text=f"전체평균({overall_avg_margin:.1%})")
        st.plotly_chart(fig_margin, use_container_width=True)
        
        fig_aov = px.bar(type_summary, x='seller_type', y='aov', title="유형별 객단가 (AOV)", color='seller_type', text_auto='.3s')
        fig_aov.add_hline(y=overall_avg_aov, line_dash="dash", line_color="red", annotation_text=f"전체평균({overall_avg_aov:,.0f}원)")
        st.plotly_chart(fig_aov, use_container_width=True)

    # [수정포인트 2] 포지셔닝 맵 사분면 가이드 추가
    st.markdown("#### 셀러별 성과 포지셔닝 맵 (Bubble Chart)")
    
    avg_order = curr_seller_metrics['order_count'].mean()
    avg_margin = curr_seller_metrics['avg_margin'].mean()
    
    fig_scatter = px.scatter(
        curr_seller_metrics.reset_index(),
        x='order_count',
        y='avg_margin',
        size='aov',
        color='seller_type',
        hover_name='셀러명',
        hover_data={'repurchase_rate': ':.1%', 'total_revenue': ':,.0f'},
        labels={'order_count': '주문 규모', 'avg_margin': '평균 마진율'},
        title=f"셀러별 주문규모 vs 마진율 (기준선: 평균 규모 {avg_order:.0f}, 평균 마진 {avg_margin:.1%})"
    )
    
    # 사분면 기준선 추가
    fig_scatter.add_hline(y=avg_margin, line_dash="dash", line_color="gray")
    fig_scatter.add_vline(x=avg_order, line_dash="dash", line_color="gray")
    
    # 사분면 텍스트 가이드
    fig_scatter.add_annotation(x=curr_seller_metrics['order_count'].max()*0.8, y=curr_seller_metrics['avg_margin'].max()*0.9, text="✨ 핵심 유지 파트너", showarrow=False, font=dict(color="green"))
    fig_scatter.add_annotation(x=curr_seller_metrics['order_count'].min()*1.2, y=curr_seller_metrics['avg_margin'].max()*0.9, text="🌱 성장·육성 대상", showarrow=False, font=dict(color="blue"))
    fig_scatter.add_annotation(x=curr_seller_metrics['order_count'].max()*0.8, y=curr_seller_metrics['avg_margin'].min()*1.1, text="⚠️ 비용·구조 관리", showarrow=False, font=dict(color="orange"))
    fig_scatter.add_annotation(x=curr_seller_metrics['order_count'].min()*1.2, y=curr_seller_metrics['avg_margin'].min()*1.1, text="🔍 점검 대상", showarrow=False, font=dict(color="red"))
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    with st.expander("💡 사분면 기반 액션 힌트 확인"):
        st.write("""
        - **고마진 / 고볼륨 (핵심 유지 파트너)**: 플랫폼 수익과 성장의 핵심. 추가적인 마케팅 구좌 지원 및 전용 프로모션 적극 검토 필요.
        - **고마진 / 저볼륨 (성장·육성 대상)**: 수익성은 좋으나 노출이 부족한 그룹. 트래픽 유입 기반의 기획전 참여 독려 시 매출 퀀텀 점프 가능성 높음.
        - **저마진 / 고볼륨 (비용·구조 관리)**: 플랫폼 활동성은 높으나 내실이 부족함. 리베이트 비율 조정이나 고단가 상품 구성을 통한 체질 개선 컨설팅 제언.
        - **저마진 / 저볼륨 (점검 대상)**: 상품 경쟁력 또는 가격 정책의 근본적 점검 필요. 신규 시즌 상품 교체나 운영 전략의 대대적 전환 권고.
        """)

# ------------------------------------------------------------------------------
# 탭 라. 상품 구조 분석 (Product Analytics)
# ------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("📦 플랫폼 상품 구조 분석")
    
    # [수정포인트 3] 인사이트 가이드 문구 추가
    premium_trend = premium_share - prev_premium_share
    headline_msg = ""
    if premium_trend > 0:
        headline_msg = f"📢 **현재 매출 성장은 프리미엄 상품 비중({premium_share:.1%}, ▲{premium_trend:.1%}) 증가에 의해 견인되고 있습니다.**"
    elif premium_trend < 0:
        headline_msg = f"📢 **프리미엄 상품 성장이 둔화({premium_share:.1%}, ▼{abs(premium_trend):.1%})세입니다. 가성비 중심의 프로모션 점검이 필요합니다.**"
    else:
        headline_msg = f"📢 **상품 구조가 안정적인 비중({premium_share:.1%})을 유지하며 내실을 다지고 있습니다.**"
        
    st.success(headline_msg)
    
    p_col1, p_col2 = st.columns(2)

    with p_col1:
        # 등급별 매출 비중
        grade_rev = filtered_df.groupby('상품성등급_그룹')['실결제 금액'].sum().reset_index()
        fig_grade = px.pie(grade_rev, values='실결제 금액', names='상품성등급_그룹', title="상품 등급별 매출 비중", hole=0.4)
        st.plotly_chart(fig_grade, use_container_width=True)

    with p_col2:
        # 가격대별 매출 및 주문 분포
        price_stats = filtered_df.groupby('가격대').agg({
            '주문번호': 'count',
            '실결제 금액': 'sum'
        }).reset_index()
        price_stats.columns = ['가격대', '주문수', '매출액']
        
        fig_price = px.bar(
            price_stats, 
            x='가격대', 
            y=['주문수', '매출액'], 
            title="가격대별 매출 및 주문 분포", 
            barmode='group',
            category_orders={"가격대": ["1만 이하", "1~3만", "3~5만", "5~10만", "10만 초과"]},
            labels={'value': '규모', 'variable': '지표'}
        )
        st.plotly_chart(fig_price, use_container_width=True)

    st.markdown("#### 일별 매출 추이 (상품 등급별)")
    daily_rev = filtered_df.groupby([filtered_df['주문일'].dt.date, '상품성등급_그룹'])['실결제 금액'].sum().reset_index()
    daily_rev.columns = ['Date', 'Grade', 'Revenue']
    fig_line = px.line(daily_rev, x='Date', y='Revenue', color='Grade', title="날짜별 데이터 추이")
    st.plotly_chart(fig_line, use_container_width=True)

# ------------------------------------------------------------------------------
# 탭 마. 지역 소비 패턴 요약
# ------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("📍 지역별 소비 패턴 요약 (Regional Insights)")
    st.info("※ 본 지역 유형은 행정구역 기준이 아닌, 실제 구매 행동(구매 단위, 가격대, 상품 성격)을 기준으로 분류한 소비 유형입니다.")
    st.markdown("수령지(광역지역)별 지표를 클러스터링하여 각 지역의 지배적인 구매 방식을 정의함.")
    
    # 클러스터별 평균 지표 비교
    cluster_summary = region_stats.groupby('region_type').agg({
        'avg_revenue': 'mean',
        'avg_qty': 'mean',
        'premium_ratio': 'mean',
        'event_ratio': 'mean',
        'single_ratio': 'mean'
    }).reset_index()
    
    r_col1, r_col2 = st.columns([1, 1.5])
    with r_col1:
        st.markdown("#### 1. 소비 지배 유형 정의")
        for idx, row in cluster_summary.iterrows():
            with st.container():
                st.write(f"**{row['region_type']}**")
                st.caption(f"평균 구매액: {row['avg_revenue']:,.0f}원 | 프리미엄 비중: {row['premium_ratio']:.1%}")
                st.markdown("---")
        
        st.info("💡 **전략 가이드**: '소량·프리미엄형' 비중이 높은 지역은 단품/고급화 전략을, '대용량·가성비형' 비중이 높은 지역은 번들/대용량 구성을 강화하는 것이 유리함.")

    with r_col2:
        st.markdown("#### 2. 클러스터별 핵심 지표 비교")
        # 지표 선택 (멀티 선택)
        metrics_to_show = st.multiselect("비교 지표 선택", ['avg_revenue', 'avg_qty', 'premium_ratio', 'event_ratio', 'single_ratio'], default=['premium_ratio', 'single_ratio', 'avg_qty'])
        
        # 긴 형태의 데이터로 변환하여 시각화
        long_cluster_summary = cluster_summary.melt(id_vars='region_type', value_vars=metrics_to_show)
        fig_cluster = px.bar(long_cluster_summary, x='region_type', y='value', color='variable', barmode='group', title="타입별 지표 비교")
        st.plotly_chart(fig_cluster, use_container_width=True)
        
    st.markdown("#### 3. 지역별 분류 상세 (Mapping)")
    st.table(region_stats[['region_type', 'avg_revenue', 'premium_ratio', 'single_ratio']].sort_values(by='region_type'))

# ------------------------------------------------------------------------------
# 탭 다. 상세 데이터 탐색 (Drill-down)
# ------------------------------------------------------------------------------
with tabs[4]:
    st.subheader("🔎 셀러 상세 성과 Drill-down")
    st.markdown("실무자가 특정 셀러의 지표를 세부적으로 조회하고 정렬할 수 있는 영역입니다.")

    display_df = curr_seller_metrics.reset_index()[['셀러명', 'seller_type', 'total_revenue', 'avg_margin', 'avg_markup', 'cancel_rate', 'repurchase_rate', 'aov']]
    display_df.columns = ['셀러명', '유형', '총 매출액(원)', '마진율(Gross)', '원가대비 마진율(Markup)', '취소율', '재구매율', '객단가(AOV)']

    # [수정포인트 4] Drill-down 테이블 우선순위 시각화
    def highlight_priority(df):
        style = df.style.format({
            '총 매출액(원)': '{:,.0f}',
            '마진율(Gross)': '{:.1%}',
            '원가대비 마진율(Markup)': '{:.1%}',
            '취소율': '{:.1%}',
            '재구매율': '{:.1%}',
            '객단가(AOV)': '{:,.0f}'
        })
        
        # 재구매율 상위 10% 하이라이트
        rep_threshold = df['재구매율'].quantile(0.9)
        style = style.apply(lambda x: ['background-color: #d1e7dd' if v >= rep_threshold else '' for v in x], subset=['재구매율'])
        
        # 취소율 10% 이상 경고 (텍스트 빨간색)
        style = style.apply(lambda x: ['color: #dc3545; font-weight: bold' if v >= 0.1 else '' for v in x], subset=['취소율'])
        
        # 성과 상위 매출 하이라이트 (배경 그래디언트)
        style = style.background_gradient(subset=['총 매출액(원)'], cmap='YlGnBu')
        
        return style

    st.dataframe(
        highlight_priority(display_df),
        use_container_width=True
    )

    with st.expander("📝 운영 의사결정 지원 포인트 요약"):
        st.write("""
        1. **종합 개요**: 플랫폼 전체의 외형 성장과 운영 효율(취소율, 마진)을 실시간으로 점검합니다.
        2. **셀러 분석**: 각 유형별 마진-규모-재구매율의 상관관계를 통해 플랫폼 내 우량 셀러를 발굴합니다. 특히 **마진율(Gross)**과 **원가대비 마진율(Markup)**을 교차 확인하여 소싱 효율을 진단합니다.
        3. **상품 분석**: 가격대 및 등급별 매출 비중을 파악하여 상품 포트폴리오를 최적화합니다.
        4. **지역 분석**: 지역별 소비 클러스터를 바탕으로 지역 특화 상품 소싱 및 배송 전략의 근거로 활용합니다.
        5. **상세 데이터**: 이슈 셀러(취소율 상위 등)를 식별하여 운영 담당자가 즉시 커뮤니케이션을 수행하는 근거로 활용합니다.
        """)
