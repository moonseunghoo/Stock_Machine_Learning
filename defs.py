import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import multiprocessing
import pandas_market_calendars as mcal
import warnings
import time
import ta

from PublicDataReader import Fred
from pykrx import stock
from marcap import marcap_data
from functools import reduce

def CPI() -> pd.DataFrame:
    """CPI(Consumer Price Index) 시리즈 데이터 조회 함수"""

    #FRED API 키    
    api_key = "8719c9b0cc99f6dda2a3ac2ae6f8a84d"

    #인스턴스 생성
    api = Fred(api_key)

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

    del_date = pd.to_datetime('2020-01-01')
    df = df[df['date'] >= del_date]
    df = df.set_index('date')

    return df

def FED_RATE() -> pd.DataFrame:
    """목표 연준 기준금리 시리즈 데이터 조회 함수"""

    #FRED API 키    
    api_key = "8719c9b0cc99f6dda2a3ac2ae6f8a84d"

    #인스턴스 생성
    api = Fred(api_key)
    
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
   
    del_date = pd.to_datetime('2020-01-01')
    df = df[df['date'] >= del_date]
    df = df.set_index('date')

    return df

#종목코드 생성
def ticker_list():
    # KOSPI 종목 코드 불러오기
    kospi = fdr.StockListing('KOSPI')
    kosdaq = fdr.StockListing('KOSDAQ')

    # 거래량이 10,000이상, 1000원 이상인 종목 필터링
    kospi = kospi[kospi['Volume'] > 1000]
    kospi = kospi[kospi['Open'] > 1000]
    
    kosdaq = kosdaq[kosdaq['Volume'] > 1000]
    kosdaq = kosdaq[kosdaq['Open'] > 1000]

    # code = pd.DataFrame(kospi['Code'])
    code = pd.concat([kospi['Code'],kosdaq['Code']],axis=0)

    code_list = [item for item in code if not any(char.isalpha() for char in item)]
    print(len(code_list))
    return code_list

def Data_Scrap():
    code_list = ticker_list()
    # code_list = ['352820']

    stock_data = []
    #보조지표 df 생성
    s_list = scrap_sub_data()
    #merging_stock_data에 입력할 2개 인자 
    input_data = [(code_list, s_list) for code_list in code_list]
    ## 멀티 프로세싱 ###
    p = multiprocessing.Pool(processes=8)
    for row in p.starmap(merging_stock_data, input_data):
        stock_data += row
    p.close()
    p.join()
    s_df = pd.DataFrame(stock_data)
    #timestamp 형식 int로 변환'
    s_df[1] = s_df[1].dt.year * 10000 + s_df[1].dt.month * 100 + s_df[1].dt.day
    s_df = s_df[~s_df[0].str.contains('K|L|M')]

    return s_df

def merging_stock_data(code, s_list):
    merge_stock_list = []
    sub_list = s_list
    stock_list = scrap_stock_data(code)
    #data열을 기준으로 2개의 데이터프레임 병합
    total_list = pd.merge(stock_list,sub_list,how='outer',on='Date')
    # total_list = stock_list
    #inf 값 Nan값으로 대체
    total_list.replace([np.inf, -np.inf], np.nan, inplace=True)
    #Nan값 없애고 리스트화
    total_list = total_list.dropna(axis=0).reset_index(drop=True).values.tolist()
    #각 행에 종목코드 추가
    for row in total_list:
        row.insert(0,code)
        merge_stock_list.append(row)
    return merge_stock_list

def m_df_to_d_df(m_df):

    start_date = '2020-01-01'
    end_date = '2024-02-01'
    date_range = pd.date_range(start_date,end_date,freq='D')
    ch_df = m_df.reindex(date_range).fillna(method='ffill')

    return ch_df

def filter_df(df):
    # 경고 메시지 무시 설정
    warnings.filterwarnings("ignore")
    # 한국 주식시장(KRX)의 개장일 캘린더 생성
    krx = mcal.get_calendar('XKRX')
    # 개장일 가져오기
    schedule = krx.schedule('2020-01-01','2024-02-01')
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
    
    stock_df = fdr.DataReader(code,'2020-01-01','2024-02-01').reset_index()

    # 이동평균선 5,20,60,200
    ma = [5,20,60,120]
    for days in ma:
        stock_df['ma_'+str(days)] = stock_df['Close'].rolling(window = days).mean().round(2)

    #52주 최고가
    stock_df['52HIGH'] = stock_df['Close'].rolling(window=252, min_periods=1).max()

    H, L, C, V = stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Volume']
    #해당 종목 EMA
    stock_df['EMA'] = ta.trend.ema_indicator(close=C, fillna=True).round(2)
    time.sleep(0.1)
    #해당 종목 SMA
    stock_df['SMA'] = ta.trend.sma_indicator(close=C, fillna=True).round(2)
    time.sleep(0.1)
    #해당 종목 RSI
    stock_df['RSI'] = ta.momentum.rsi(close=C, fillna=True).round(2)
    time.sleep(0.1)
    #해당 종목 MFI
    stock_df['MFI'] = ta.volume.money_flow_index(high=H, low=L, close=C, volume=V, fillna=True).round(2)
    time.sleep(0.1)
    #해당 종목 상승률
    stock_df['Change'] = (stock_df['Change']*100).round(2)
    #해당 종목 VPT
    stock_df['VPT'] = ta.volume.volume_price_trend(close=C, volume=V).round(2)
    time.sleep(0.1)
    #해당 종목 일목균형표
    stock_df['VI'] = ta.trend.vortex_indicator_pos(high=H,low=L,close=C,fillna=True).round(2)
    time.sleep(0.1)
    #해당 종목 Bolinger Bend
    stock_df['BB'] = ta.volatility.bollinger_hband(close=C,window=20,window_dev=2,fillna=True).round(2)
    time.sleep(0.1)
    #해당 종목 MACD
    stock_df['MACD_L'] = ta.trend.MACD(close=C,window_fast=12,window_slow=26,window_sign=9,fillna=False).macd().round(2)
    time.sleep(0.05)
    stock_df['MACD_S'] = ta.trend.MACD(close=C,window_fast=12,window_slow=26,window_sign=9,fillna=False).macd_signal().round(2)
    time.sleep(0.05)
    #해당 종목 SR
    stock_df['SR'] = ta.momentum.StochasticOscillator(close=C,high=H,low=L,window=14,smooth_window=3,fillna=False).stoch().round(2)
    time.sleep(0.05)
    stock_df['SR_S'] = ta.momentum.StochasticOscillator(close=C,high=H,low=L,window=14,smooth_window=3,fillna=False).stoch_signal().round(2)
    time.sleep(0.05)

    #해당종목 WR
    stock_df['WR'] = ta.momentum.WilliamsRIndicator(high=H,low=L,close=C,lbp=14,fillna=False).williams_r().round(2)

    #y 라벨 결과 데이터프레임 
    # stock_df['Label'] = (stock_df['Change'] >= 5)
    label_df = pd.DataFrame()
    label_df['전일종가'] = stock_df['Close'].shift(1)
    label_df['고가변동'] = (stock_df['High'] - label_df['전일종가']) / label_df['전일종가'] * 100

    # '고가변동' 열이 3% 이상인 경우 'Label' 열에 True를 추가합니다.
    stock_df['Label'] = label_df['고가변동'] >= 3.3

    #해당 종목 시가총액, 거래대금, 주식수
    M_df = marcap_data('2020-01-01','2024-02-01',code=code)
    selected_colums = ['Marcap','Amount','Stocks']
    M_df = M_df[selected_colums].reset_index()

    #데이터 병합
    merge_df = [stock_df,M_df]
    dataset_df = reduce(lambda x,y : pd.merge(x,y,on='Date'),merge_df)

    #주식시장 개장일만 분류
    filtered_df = filter_df(dataset_df)

    return filtered_df

def scrap_sub_data():
    warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

    # 코스피 지수 
    KSI_df = fdr.DataReader('KS11','2020-01-01','2024-02-01').reset_index().drop(
        ['Open','High','Low','Adj Close'], axis=1).rename(
            columns={'Close':'KSI_Clo','Volume':'KSI_Vol'}).round(2)
   
    # 주식시장 개장일만 분류
    filtered_df = filter_df(KSI_df)
    return filtered_df

if __name__ == '__main__':
    CPI()
    FED_RATE()
    ticker_list()
    Data_Scrap()
    merging_stock_data()
    m_df_to_d_df()
    filter_df()
    scrap_stock_data()
    scrap_sub_data()