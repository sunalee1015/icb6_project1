import pandas as pd
import numpy as np

def analyze_seller_typology(file_path):
    """
    셀러 운영 유형 정량적 정의서에 기반하여 셀러를 분류하고 성과를 요약함.
    """
    # 1. 데이터 로드 및 기초 지표 산출
    df = pd.read_excel(file_path)
    
    # 마진율 계산 (Row level)
    df['margin_ratio'] = (df['판매단가'] - df['공급단가']) / df['판매단가']
    
    # 2. 셀러 단위 집계
    seller_metrics = df.groupby('셀러명').agg({
        '상품성등급_그룹': lambda x: (x == '프리미엄').mean(),
        '이벤트 여부': lambda x: (x == 'Y').mean(),
        '주문번호': 'count',
        '판매단가': 'mean',
        '실결제 금액': 'sum',
        'margin_ratio': 'mean'
    }).rename(columns={
        '상품성등급_그룹': 'premium_ratio',
        '이벤트 여부': 'event_ratio',
        '주문번호': 'order_count',
        '판매단가': 'avg_price',
        '실결제 금액': 'total_revenue',
        'margin_ratio': 'avg_margin'
    })

    # 3. 유형 분류 로직 (우선순위 반영)
    def classify_seller(row):
        # [1순위] 프리미엄 집중형
        if row['premium_ratio'] >= 0.3:
            return '프리미엄 집중형'
        # [2순위] 이벤트 의존형
        elif row['event_ratio'] >= 0.3:
            return '이벤트 의존형'
        # [3순위] 박리다매 운영형
        elif row['order_count'] >= 15 and row['avg_price'] <= 28000:
            return '박리다매 운영형'
        # [기본] 일반 운영형
        else:
            return '일반 운영형'

    seller_metrics['seller_type'] = seller_metrics.apply(classify_seller, axis=1)

    # 4. 유형별 요약 통계 생성
    summary = seller_metrics.groupby('seller_type').agg({
        'seller_type': 'count',
        'total_revenue': 'sum',
        'avg_margin': 'mean'
    }).rename(columns={'seller_type': 'seller_count'})
    
    # 매출 비중 산출
    total_rev_sum = summary['total_revenue'].sum()
    summary['revenue_share'] = summary['total_revenue'] / total_rev_sum
    
    return seller_metrics, summary

if __name__ == "__main__":
    target_file = r"C:\data_workspace\ICB6_Project1\preprocessed_data_2026.xlsx"
    metrics, summary_report = analyze_seller_typology(target_file)
    
    print("\n[셀러 유형별 요약 보고서]")
    # 출력 포맷 조정 (퍼센트 등)
    formatted_summary = summary_report.copy()
    formatted_summary['revenue_share'] = formatted_summary['revenue_share'].map(lambda x: f"{x:.2%}")
    formatted_summary['avg_margin'] = formatted_summary['avg_margin'].map(lambda x: f"{x:.2%}")
    formatted_summary['total_revenue'] = formatted_summary['total_revenue'].map(lambda x: f"{x:,.0f}원")
    
    print(formatted_summary)
    
    print("\n[상위 10개 셀러 분류 샘플]")
    print(metrics[['seller_type', 'premium_ratio', 'event_ratio', 'order_count', 'avg_price']].head(10))
