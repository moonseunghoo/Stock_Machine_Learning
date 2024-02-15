import FinanceDataReader as fdr
import multiprocessing
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
import warnings
import time
import ta

from marcap import marcap_data
from pykrx import stock
from PublicDataReader import Fred
from datetime import datetime, timedelta
from functools import reduce

#날자 변수 생성
def date_info():
    targer_day = datetime(year=2024,month=2,day=16)

    if targer_day.strftime('%a') == 'Mon':
        end_info_day = (targer_day - timedelta(days=3))
        delta = timedelta(days=210)
        day_120 = (end_info_day - delta).strftime("%Y-%m-%d")

    elif targer_day.strftime('%a') == 'Tue':
        end_info_day = (targer_day - timedelta(days=1))
        delta = timedelta(days=210)
        day_120 = (end_info_day - delta).strftime("%Y-%m-%d")

    else :
        end_info_day = (targer_day - timedelta(days=1))
        delta = timedelta(days=210)
        day_120 = (end_info_day - delta).strftime("%Y-%m-%d")

    return targer_day, end_info_day, day_120

def add_52_week_high_info(series):
    df = pd.DataFrame(series, columns=['Code'])
    high_52_week_list = []
    change_percentage_list = []
    average_volume_5_days = []

    for stock_code in series:
        # 오늘 날짜
        # end_date = datetime.today()
        end_date = datetime(year=2024, month=2, day=15)
        # 10일 전 날짜 계산
        start_date = (end_date - timedelta(days=10)).strftime('%Y-%m-%d')

        # FinanceDataReader를 사용하여 주식 데이터 가져오기
        stock_data = fdr.DataReader(stock_code, start=start_date, end=end_date)
        stock_data = stock_data.tail(5)
        
        # 52주 최고가 계산
        fifty_two_week_high = stock_data['Close'].rolling(window=252, min_periods=1).max()
        recent_52_week_high = fifty_two_week_high.iloc[-1]
    
        # 최근 종가 구하기
        recent_close = stock_data['Close'].iloc[-1]
        
        # 최근 종가 대비 52주 최고가 대비 변동율 계산
        change_percentage = ((recent_close - recent_52_week_high) / recent_52_week_high) * 100

        # 거래량(거래대금)의 평균 계산
        average_volume = int(stock_data['Volume'].mean())
        
        high_52_week_list.append(recent_52_week_high)
        change_percentage_list.append(change_percentage)
        average_volume_5_days.append(average_volume)

    df['52주 최고가'] = high_52_week_list
    df['52주 최고가 대비 변동율'] = change_percentage_list
    df['5일 평균 거래량'] = average_volume_5_days

    return df

#종목코드 생성
def ticker_list():
    # kospi = fdr.StockListing('KOSPI')
    # kospi = kospi['Code']
    # kosdaq = fdr.StockListing('KOSDAQ')
    # kosdaq = kosdaq['Code']
    # code = pd.concat([kospi,kosdaq],axis=0)
    a = stock.get_market_ticker_list("20240215", market="KOSPI")
    b = stock.get_market_ticker_list("20240215", market="KOSDAQ")
    kospi = pd.Series(a)
    kosdaq = pd.Series(b)
    code = pd.concat([kospi,kosdaq],axis=0)

    # 데이터프레임 생성 
    result_df = add_52_week_high_info(code)

    result_df = result_df[result_df['5일 평균 거래량'] >= 500000]

    # 변동율을 기준으로 내림차순 정렬
    sorted_df = result_df.sort_values(by='52주 최고가 대비 변동율', ascending=False)

    # 상위 100개만 선택하여 새로운 데이터프레임 생성
    top_100_df = sorted_df.head(100)
    top_100_df = top_100_df.drop(['52주 최고가','52주 최고가 대비 변동율','5일 평균 거래량'],axis=1)
    top_100_list = top_100_df['Code'].tolist()

    return top_100_list

def Marcap():
    marcap = pd.read_csv('/Users/moon/Desktop/Moon SeungHoo/Stock_Machine_Learning/KRX/marcap/240215.csv', low_memory=False, encoding='euc-kr')
    # 거래량과 종가가 조건을 충족하지 못하는 종목 필터링
    marcap = marcap[marcap['거래량'] >= 500000]
    marcap = marcap[(marcap['종가'] > 5000) & (marcap['종가'] <= 50000)]
    marcap = marcap.drop(['종목명','시장구분','소속부','종가','대비','등락률','시가','고가','저가','거래량'],axis=1)
    
    return marcap

def CPI(end_info_day, day_120) -> pd.DataFrame:
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

    del_date = pd.to_datetime(day_120)
    df = df[df['date'] >= del_date]
    df = df.set_index('date')

    return df

def FED_RATE(end_info_day, day_120) -> pd.DataFrame:
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
   
    del_date = pd.to_datetime(day_120)
    df = df[df['date'] >= del_date]
    df = df.set_index('date')

    return df
 
def Data_Scrap_Pred():
    #종목코드 생성
    code_list = ticker_list() 
    # code_list = ['078890'] 
    #날자 변수 생성
    targer_day, end_info_day, day_120 = date_info()

    stock_data = []

    #marcap 
    marcap = Marcap()
    #보조지표 생성
    s_list = scrap_sub_data(end_info_day, day_120)
    #merging_stock_data에 입력할 2개 인자 
    input_data = [(code_list,marcap, s_list, end_info_day, day_120) for code_list in code_list]
    ## 멀티 프로세싱 ###
    p = multiprocessing.Pool(processes=8)
    for row in p.starmap(merging_stock_data, input_data):
        stock_data += row
    p.close()
    p.join()
    s_df = pd.DataFrame(stock_data)
    #timestamp 형식 int로 변환
    s_df[1] = s_df[1].dt.year * 10000 + s_df[1].dt.month * 100 + s_df[1].dt.day
    s_df = s_df[~s_df[0].str.contains('K|L|M')]
    
    return s_df

def merging_stock_data(code,marcap, s_list, end_info_day, day_120):
    merge_stock_list = []
    sub_list = s_list
    stock_list = scrap_stock_data(code,marcap, end_info_day, day_120)
    #data열을 기준으로 2개의 데이터프레임 병합
    total_list = pd.merge(stock_list,sub_list,how='outer',on='Date')
    # total_list = stock_list
    #inf 값 Nan으로 대체
    total_list.replace([np.inf, -np.inf], np.nan, inplace=True)
    #Nan값 없애고 리스트화
    total_list = total_list.dropna(axis=0).reset_index(drop=True).tail(1).values.tolist()
    for row in total_list:
        row.insert(0,code)
        merge_stock_list.append(row)
    return merge_stock_list

def m_df_to_d_df(m_df, end_info_day, day_120):

    start_date = day_120
    end_date = end_info_day
    date_range = pd.date_range(start_date,end_date,freq='D')
    ch_df = m_df.reindex(date_range).fillna(method='ffill')

    return ch_df

def filter_df(df, end_info_day, day_120):
    # 경고 메시지 무시 설정
    warnings.filterwarnings("ignore")
    # 한국 주식시장(KRX)의 개장일 캘린더 생성
    krx = mcal.get_calendar('XKRX')
    # 개장일 가져오기
    schedule = krx.schedule(day_120,end_info_day)
    # break_start,break_end 제거
    krx.remove_time(market_time='break_start')
    krx.remove_time(market_time='break_end')
    # 개장일에 해당하는 날짜만 추출
    market_open_dates = schedule.index
    # 데이터프레임에서 개장일에 해당하는 행만 남기기
    filter_df = df[df['Date'].isin(market_open_dates)]

    return filter_df

def scrap_stock_data(code,marcap, end_info_day, day_120):
    warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

    stock_df = fdr.DataReader(code,day_120,end_info_day).reset_index()

    # 이동평균선 5,20,60,200 O
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
    stock_df['BB'] = ta.volatility.bollinger_hband(close=C,window=7,window_dev=2,fillna=True).round(2)
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

    #거래대금 시가총액 주식수 
    M_df = marcap[marcap['종목코드'] == code].drop(['종목코드'],axis =1)
    M_extend_df = pd.concat([M_df] * (len(stock_df) // 1) + [M_df.iloc[:len(stock_df) % 1]], ignore_index=True)

    # 두 데이터프레임을 수평방향으로 병합합니다.
    result_df = pd.concat([stock_df, M_extend_df], axis=1)

    #주식시장 개장일만 분류
    filtered_df = filter_df(result_df, end_info_day, day_120)

    return filtered_df

def scrap_sub_data(end_info_day, day_120):
    warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거
    # 코스피 지수 
    kospi = pd.read_csv('/Users/moon/Desktop/Moon SeungHoo/Stock_Machine_Learning/KRX/kospi/240215.csv', low_memory=False, encoding='euc-kr')
    kospi = kospi.drop(['대비','등락률','시가','고가','저가','거래대금','상장시가총액'],axis=1)

    rows = kospi[kospi['지수명'] == '코스피'].rename(columns={'지수명': 'Date'})
    rows['거래량'] = rows['거래량'].astype(int)
    rows = rows.replace({'Date' : '코스피'}, pd.to_datetime('2024-02-15'))

    filtered_df = filter_df(rows, end_info_day, day_120)
    return filtered_df

if __name__ == '__main__':
    date_info()
    add_52_week_high_info()
    ticker_list()
    Marcap()
    CPI()
    FED_RATE()
    Data_Scrap_Pred()
    merging_stock_data()
    m_df_to_d_df()
    filter_df()
    scrap_stock_data()
    scrap_sub_data()
