import pandas as pd
import numpy as np
import json

def analyze_performance_decomposition(file_path):
    df = pd.read_excel(file_path)
    
    # 지표 산출
    df['마진율'] = (df['판매단가'] - df['공급단가']) / df['판매단가']
    df['취소여부_num'] = df['취소여부'].apply(lambda x: 1 if x == 'Y' else 0)
    
    # 1) 동일 셀러 내 상품 성격별 성과 차이
    # 주문수가 일정 수준(예: 30건) 이상인 셀러만 대상으로 분석의 신뢰성 확보
    seller_counts = df['셀러명'].value_counts()
    valid_sellers = seller_counts[seller_counts >= 30].index
    df_valid = df[df['셀러명'].isin(valid_sellers)]
    
    within_analysis = df_valid.groupby(['셀러명', '상품성등급_그룹', '이벤트 여부']).agg({
        '마진율': 'mean',
        '취소여부_num': 'mean',
        '실결제 금액': 'mean'
    }).reset_index()
    
    # 2) 셀러 간 성과 차이 요인 (상품 믹스 vs 운영 역량)
    seller_stats = df.groupby('셀러명').agg({
        '상품성등급_그룹': lambda x: x.value_counts(normalize=True).to_dict(),
        '이벤트 여부': lambda x: (x == 'Y').mean(),
        '마진율': 'mean',
        '취소여부_num': 'mean',
        '실결제 금액': 'mean'
    }).rename(columns={'이벤트 여부': '이벤트_비중', '실결제 금액': '주문당_평균매출'})
    
    # 상품 등급별(믹스별) 전체 평균 성과 (기준점)
    baseline_stats = df.groupby('상품성등급_그룹').agg({
        '마진율': 'mean',
        '취소여부_num': 'mean',
        '실결제 금액': 'mean'
    }).to_dict(orient='index')
    
    # 3) 셀러 유형화 기초 (분포 확인)
    type_indicators = df.groupby('셀러명').agg({
        '이벤트 여부': lambda x: (x == 'Y').mean(),
        '판매단가': 'mean',
        '상품성등급_그룹': lambda x: (x == '프리미엄').mean(),
        '주문번호': 'count'
    }).rename(columns={'이벤트 여부': '이벤트_의존도', '판매단가': '평균_판매가', '상품성등급_그룹': '프리미엄_비중', '주문번호': '규모'})
    
    # JSON 직렬화를 위한 정리
    result = {
        "within_seller_stats": within_analysis.head(50).to_dict(orient='records'),
        "baseline_stats": baseline_stats,
        "type_indicators_summary": type_indicators.describe().to_dict(),
        "event_impact_overall": df.groupby('이벤트 여부').agg({'마진율': 'mean', '취소여부_num': 'mean'}).to_dict(orient='index')
    }
    
    print(json.dumps(result, indent=4, default=str))

if __name__ == "__main__":
    analyze_performance_decomposition(r"C:\data_workspace\ICB6_Project1\preprocessed_data_2026.xlsx")
