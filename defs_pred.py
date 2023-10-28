import FinanceDataReader as fdr
import pandas as pd
import pandas_market_calendars as mcal
import warnings
import ta

from pykrx import stock
from marcap import marcap_data
from PublicDataReader import Fred
from datetime import datetime, timedelta
from functools import reduce

#FRED API 키    
api_key = "8719c9b0cc99f6dda2a3ac2ae6f8a84d"
#오늘 날자 가져오기
today = '2023-10-13'
today_ = datetime.today()
delta = timedelta(days=210)
day_120 = (today_ - delta).strftime("%Y-%m-%d")
today_ = today_.strftime("%Y-%m-%d")


#당일 날자까지 빈 날자 채우기 위한 값
specific_value = pd.to_datetime(today)

#인스턴스 생성
api = Fred(api_key)

def CPI() -> pd.DataFrame:
    """CPI(Consumer Price Index) 시리즈 데이터 조회 함수"""
    # 시리즈 ID값
    series_id = "CPIAUCNS"

    #시리즈 데이터 조회
    df = api.get_data(
        api_name="series_observations",
        series_id = series_id
    )

    # value 컬럼 숫자형 변환
    df['value'] = pd.to_numeric(df['value'], errors="coerce")
    # date 컬럼 날짜형 변환
    df['date'] = pd.to_datetime(df['date'])
    # date 컬럼을 인덱스로 설정
    df = df.set_index("date")

    # 전년 동월 값 컬럼 생성
    df['value_last_year'] = df['value'].shift(12)
    df['CPI(YoY)'] = (df['value'] - df['value_last_year']) / df['value_last_year'] * 100
    
    # 전년 동월 값 컬럼만 선택
    df = df[['CPI(YoY)']].reset_index()

    del_date = pd.to_datetime(day_120)
    df = df[df['date'] >= del_date]
    df = df.set_index('date')

    return df

def FED_RATE() -> pd.DataFrame:
    """목표 연준 기준금리 시리즈 데이터 조회 함수"""
    
    # 시리즈 ID 값
    series_id = "DFEDTARU"
    
    # 시리즈 데이터 조회
    df = api.get_data(
        api_name="series_observations", 
        series_id=series_id
    )
    
    # value 컬럼 숫자형 변환 컬럼 생성
    df['FED RATE'] = pd.to_numeric(df['value'], errors="coerce")
    # date 컬럼 날짜형 변환
    df['date'] = pd.to_datetime(df['date'])

     # date 컬럼을 인덱스로 설정
    df = df.set_index('date')
    
    # 기준금리 컬럼만 선택
    df = df[['FED RATE']].reset_index()
   
    del_date = pd.to_datetime(day_120)
    df = df[df['date'] >= del_date]
    df = df.set_index('date')

    return df
 
def merging_stock_data(code):
    merge_stock_list = []
    stock_list = scrap_stock_data(code)
    total_list = pd.merge(stock_list,sub_list,how='outer',on='Date')
    total_list = total_list.dropna(axis=0).reset_index(drop=True).tail(3).values.tolist()
    for row in total_list:
        row.insert(0,code)
        merge_stock_list.append(row)
    return merge_stock_list

def m_df_to_d_df(m_df):

    start_date = day_120
    end_date = today
    date_range = pd.date_range(start_date,end_date,freq='D')
    ch_df = m_df.reindex(date_range).fillna(method='ffill')

    return ch_df

def filter_df(df):
    # 경고 메시지 무시 설정
    warnings.filterwarnings("ignore")
    # 한국 주식시장(KRX)의 개장일 캘린더 생성
    krx = mcal.get_calendar('XKRX')
    # 개장일 가져오기
    schedule = krx.schedule(day_120,today)
    # break_start,break_end 제거
    krx.remove_time(market_time='break_start')
    krx.remove_time(market_time='break_end')
    # 개장일에 해당하는 날짜만 추출
    market_open_dates = schedule.index
    # 데이터프레임에서 개장일에 해당하는 행만 남기기
    filter_df = df[df['Date'].isin(market_open_dates)]

    return filter_df

def scrap_stock_data(code):
    warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

    stock_df = fdr.DataReader(code,day_120,today).reset_index()

    buy_df = stock.get_market_trading_volume_by_date(day_120,today, code, on='매수').iloc[:,2].reset_index()
    sell_df = stock.get_market_trading_volume_by_date(day_120,today, code, on='매도').iloc[:,2].reset_index()

    # 이동평균선 5,20,60,200 O
    ma = [5,20,60,120]
    for days in ma:
        stock_df['ma_'+str(days)] = stock_df['Close'].rolling(window = days).mean().round(2)

    H, L, C, V = stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Volume']
    #해당 종목 EMA
    stock_df['EMA'] = ta.trend.ema_indicator(close=C, fillna=True).round(2)
    #해당 종목 SMA
    stock_df['SMA'] = ta.trend.sma_indicator(close=C, fillna=True).round(2)
    #해당 종목 RSI
    stock_df['RSI'] = ta.momentum.rsi(close=C, fillna=True).round(2)
    #해당 종목 MFI
    stock_df['MFI'] = ta.volume.money_flow_index(high=H, low=L, close=C, volume=V, fillna=True).round(2)
    #해당 종목 상승률
    stock_df['Change'] = (stock_df['Change']*100).round(2)
    #해당 종목 VPT
    stock_df['VPT'] = ta.volume.volume_price_trend(close=C, volume=V).round(2)
    #해당 종목 일목균형표
    stock_df['VI'] = ta.trend.vortex_indicator_pos(high=H,low=L,close=C,fillna=True).round(2)
    #해당 종목 체결강도
    stock_df['VP'] = (buy_df['개인']/sell_df['개인']*100).round(2)

    #해당 종목 시가총액, 거래대금, 주식수
    M_df = marcap_data(day_120,today,code=code)
    selected_colums = ['Amount','Marcap','Stocks']
    M_df = M_df[selected_colums].reset_index()

    #데이터 병합
    merge_df = [stock_df,M_df]
    dataset_df = reduce(lambda x,y : pd.merge(x,y,on='Date'),merge_df)

    #주식시장 개장일만 분류
    filtered_df = filter_df(dataset_df)
    return filtered_df

def scrap_sub_data():
    # 다우존스 지수 
    warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거
    # DJI_df = fdr.DataReader('DJI',day_120,today).reset_index().drop(
    #     ['Open','High','Low','Adj Close'], axis=1).rename(columns={
    #         'Close':'DJI_Clo','Volume':'DJI_Vol'}).round(2)

    # 나스닥 지수
    IXIC_df = fdr.DataReader('IXIC',day_120,today).reset_index().drop(
        ['Open','High','Low','Adj Close'], axis=1).rename(
            columns={'Close':'IXIC_Clo','Volume':'IXIC_Vol'}).round(2)

    # # S&P500 지수 
    # SP5_df = fdr.DataReader('US500',day_120,today).reset_index().drop(
    #     ['Open','High','Low','Adj Close'], axis=1).rename(columns={
    #         'Close':'SP5_Clo','Volume':'SP5_Vol'}).round(2)

    # VIX 지수 
    VIX_df = fdr.DataReader('VIX',day_120,today).reset_index().drop(
        ['Open','High','Low','Volume','Adj Close'], axis=1).rename(
            columns={'Close':'VIX_Clo'}).round(2)


    # 코스피 지수 
    KSI_df = fdr.DataReader('KS11',day_120,today).reset_index().drop(
        ['Open','High','Low','Adj Close'], axis=1).rename(columns={
            'Close':'KSI_Clo','Volume':'KSI_Vol'}).round(2)

    # 원/달러 환율 
    USD_df = fdr.DataReader('USD/KRW',day_120,today).reset_index().drop(
        ['Open','High','Low','Adj Close','Volume'], axis=1).rename(columns={
            'Close':'USD/KRW_CLO'}).round(2)

    # # 미국 국채 금리 (20,10,5,1년) 
    # DGS_df = fdr.DataReader('FRED:DGS20,DGS10,DGS5,DGS1', day_120
    #                         ).reset_index().rename(columns={'DATE':'Date'}).round(2)

    #미국 소비자심리 지수(CSI) 
    UMCSENT_df_b = fdr.DataReader('FRED:UMCSENT', day_120)
    #M2 통화량 
    M2SL_df_b = fdr.DataReader('FRED:M2SL', day_120)
    #CPI 지표
    CPI_df_b = CPI()
    #연준 기준금리
    FDR_df = FED_RATE().reset_index().rename(columns={'date':'Date'}).round(2)

    #월단위 데이터 일단위로 변환
    CPI_df = m_df_to_d_df(CPI_df_b).reset_index().rename(columns={'index':'Date'}).round(2)
    M2SL_df = m_df_to_d_df(M2SL_df_b).reset_index().rename(columns={'index':'Date'}).round(2)
    UMCSENT_df = m_df_to_d_df(UMCSENT_df_b).reset_index().rename(columns={'index':'Date'}).round(2)

    # #데이터프레임 병합
    data_df = [IXIC_df,VIX_df,KSI_df,USD_df,UMCSENT_df,M2SL_df,CPI_df,FDR_df]
    dataset_df = reduce(lambda x,y : pd.merge(x,y,on='Date'),data_df)

    # 주식시장 개장일만 분류
    filtered_df = filter_df(dataset_df)
    return filtered_df

def add_row(list):
    # 데이터프레임의 'Date' 열의 최하단 값 가져오기
    last_date = list['Date'].iloc[-1]

    # 최하단 값과 특정 값 비교
    while last_date != specific_value:
        # 'Date' 열을 하루씩 늘리기
        last_date = last_date + timedelta(days=1)
        
        # 나머지 열의 최하단 값을 복사하여 새로운 행 추가
        new_row = {'Date': last_date}
        for col in list.columns:
            if col != 'Date':
                new_row[col] = list[col].iloc[-1]
        
        # 새로운 행을 데이터프레임에 추가
        list = pd.concat([list, pd.DataFrame(new_row, index=[0])], ignore_index=True)
    return list

sub_list = scrap_sub_data()