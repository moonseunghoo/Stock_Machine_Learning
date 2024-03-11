import FinanceDataReader as fdr
import multiprocessing
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
import warnings
import time
import ta

from pykrx import stock
from PublicDataReader import Fred
from datetime import datetime, timedelta

#날자 변수 생성
def date_info():
    today = datetime.now()
    # today = (datetime.now() - timedelta(days=3))

    if today.strftime('%a') == 'Fri':
        targer_day = (today + timedelta(days=3))
    else :
        targer_day = (today + timedelta(days=1))

    if targer_day.strftime('%a') == 'Mon':
        end_info_day = (targer_day - timedelta(days=3))
        delta = timedelta(days=375) 
        delta2 = timedelta(days=60)
        day_21 = (end_info_day - delta2).strftime("%Y-%m-%d")
        day_120 = (end_info_day - delta).strftime("%Y-%m-%d")
    else :
        end_info_day = (targer_day - timedelta(days=1))
        delta = timedelta(days=375)
        delta2 = timedelta(days=60)
        day_21 = (end_info_day - delta2).strftime("%Y-%m-%d")
        day_120 = (end_info_day - delta).strftime("%Y-%m-%d")

    return targer_day, end_info_day, day_120, day_21

def add_52_week_high_info(series, end_info_day):
    df = pd.DataFrame(series, columns=['Code'])
    high_52_week_list = []
    change_percentage_list = []
    average_volume_5_days = []

    for stock_code in series:
    # 10일 전 날짜 계산
        high_52_date = (end_info_day - timedelta(days=375)).strftime('%Y-%m-%d')

        # FinanceDataReader를 사용하여 주식 데이터 가져오기
        stock_data = fdr.DataReader(stock_code, start=high_52_date, end=end_info_day).reset_index()
        stock_data['Volume'] = stock_data['Volume'] * ((stock_data['High'] + stock_data['Low']) / 2)

        # 52주 최고가 계산
        rolling_result = stock_data['High'].rolling(window=250).max()
        
        # 최근 종가 대비 52주 최고가 대비 변동율 계산
        change_percentage = round((stock_data['Close'].shift(1) - rolling_result) / rolling_result, 3)

        # 거래대금 평균 계산
        stock_data = stock_data.tail(5)
        average_volume = int(stock_data['Volume'].mean())
        
        high_52_week_list.append(rolling_result.tail(1))
        change_percentage_list.append(change_percentage.tail(1))
        average_volume_5_days.append(average_volume)

    df['52주 최고가'] = high_52_week_list
    df['52주 최고가 대비 변동율'] = change_percentage_list
    df['5일 평균 거래대금'] = average_volume_5_days

    return df

#종목코드 생성
def ticker_list(end_info_day):
    a = stock.get_market_ticker_list(market="KOSPI")
    b = stock.get_market_ticker_list(market="KOSDAQ")
    kospi = pd.Series(a)
    kosdaq = pd.Series(b)
    code = pd.concat([kospi,kosdaq],axis=0)

    # 데이터프레임 생성 
    result_df = add_52_week_high_info(code, end_info_day)

    result_df = result_df[result_df['5일 평균 거래대금'] >= 45000000000]
    # result_df['52주 최고가 대비 변동율'] = result_df['52주 최고가 대비 변동율'].apply(lambda x: float(x))

    # 변동율을 기준으로 내림차순 정렬
    # '52주 최고가 대비 변동율' 열의 값을 float로 변환
    # result_df['52주 최고가 대비 변동율'] = result_df['52주 최고가 대비 변동율'].astype(float)
    result_df['52주 최고가 대비 변동율'] = result_df['52주 최고가 대비 변동율'].apply(lambda x: float(x.iloc[0]))

    sorted_df = result_df.sort_values(by='52주 최고가 대비 변동율', ascending=True)

    # 상위 100개만 선택하여 새로운 데이터프레임 생성
    top_100_df = sorted_df.head(100)
    top_100_df = top_100_df.drop(['52주 최고가','52주 최고가 대비 변동율','5일 평균 거래대금'],axis=1)
    top_100_list = top_100_df['Code'].tolist()

    return top_100_list

def Marcap():
    # date = (datetime.now() - timedelta(days=3)).strftime('%Y%m%d')[2:]
    date = datetime.now().strftime('%Y%m%d')[2:]
    marcap = pd.read_csv('KRX/marcap/' + date + '.csv', encoding='euc-kr')
    # 거래량과 종가가 조건을 충족하지 못하는 종목 필터링
    marcap = marcap[(marcap['종가'] > 5000)]
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
    #날자 변수 생성
    targer_day, end_info_day, day_120, day_21 = date_info()
    #종목코드 생성
    code_list = ticker_list(end_info_day) 
    # code_list = ['222080'] 

    stock_data = []
    #marcap 
    marcap = Marcap()
    #보조지표 생성
    s_list = scrap_sub_data(end_info_day, day_120, day_21)
    #merging_stock_data에 입력할 2개 인자 
    input_data = [(code_list,marcap, s_list, end_info_day, day_120) for code_list in code_list]
    ## 멀티 프로세싱 ###
    p = multiprocessing.Pool(processes=8)
    for row, column_names in p.starmap(merging_stock_data, input_data):
        stock_data += row
    p.close()
    p.join()
    column_names.insert(0,'Ticker')
    s_df = pd.DataFrame(stock_data, columns=column_names)
    #timestamp 형식 int로 변환
    # s_df['Date'] = s_df['Date'].dt.year * 10000 + s_df['Date'].dt.month * 100 + s_df['Date'].dt.day
    s_df = s_df[~s_df['Ticker'].str.contains('K|L|M')]
    
    return s_df

def merging_stock_data(code,marcap, s_list, end_info_day, day_120):
    merge_stock_list = []
    sub_list = s_list
    stock_list = scrap_stock_data(code,marcap, end_info_day, day_120)
    #data열을 기준으로 2개의 데이터프레임 병합
    total_list = pd.merge(stock_list,sub_list,how='outer',on='Date')
    #inf 값 Nan으로 대체
    total_list.replace([np.inf, -np.inf], np.nan, inplace=True)
    column_names = total_list.columns.tolist()
    #Nan값 없애고 리스트화
    total_list = total_list.dropna(axis=0).reset_index(drop=True).tail(1).values.tolist()
    for row in total_list:
        row.insert(0,code)
        merge_stock_list.append(row)
    return merge_stock_list, column_names

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
    stock_df['52High'] = stock_df['High'].rolling(window=250).max()

    stock_df['52Change'] = round((stock_df['Close'].shift(1) - stock_df['52High']) / stock_df['52High'], 3)

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

def scrap_sub_data(end_info_day, day_120, day_21):
    warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거
    
    KSI_df = fdr.DataReader('KS11', day_21, end_info_day).reset_index().round(2)
    KSI_df.drop(KSI_df.columns[-2], axis=1, inplace=True)

    date = end_info_day.strftime("%Y-%m-%d").replace("-", "")[2:]
    s_date = end_info_day.strftime("%Y-%m-%d")

    # 코스피 지수 
    kospi = pd.read_csv('KRX/kospi/' + date + '.csv', 
                        encoding='euc-kr').drop(['대비','등락률','거래대금','상장시가총액'],axis=1).rename(
                        columns={'시가':'Open','고가':'High','저가':'Low','종가':'Close','거래량':'Volume'})
    rows = kospi[kospi['지수명'] == '코스피'].rename(columns={'지수명': 'Date'})
    rows['Volume'] = rows['Volume'].astype(int)
    rows = rows.replace({'Date' : '코스피'}, pd.to_datetime(s_date))

    new_order = ['Date', 'Open', 'High','Low','Close','Volume']  # 새로운 열 순서 지정
    rows = rows[new_order]

    df = pd.concat([KSI_df,rows]).drop(labels='Adj Close', axis=1)

    H, L, C, V = df['High'], df['Low'], df['Close'], df['Volume']

    #해당 종목 RSI
    df['RSI'] = ta.momentum.rsi(close=C, window=14, fillna=True).round(2)

    #해당 종목 MACD
    df['MACD_L'] = ta.trend.MACD(close=C,window_fast=12,window_slow=26,window_sign=9,fillna=False).macd().round(2)
    time.sleep(0.05)
    df['MACD_S'] = ta.trend.MACD(close=C,window_fast=12,window_slow=26,window_sign=9,fillna=False).macd_signal().round(2)

    #해당 종목 Bolinger Bend
    df['BB'] = ta.volatility.bollinger_hband(close=C,window=7,window_dev=2,fillna=True).round(2)

    #해당 종목 SR
    df['SR'] = ta.momentum.StochasticOscillator(close=C,high=H,low=L,window=14,smooth_window=3,fillna=False).stoch().round(2)
    time.sleep(0.05)
    df['SR_S'] = ta.momentum.StochasticOscillator(close=C,high=H,low=L,window=14,smooth_window=3,fillna=False).stoch_signal().round(2)

    #해당종목 WR
    df['WR'] = ta.momentum.WilliamsRIndicator(high=H,low=L,close=C,lbp=14,fillna=False).williams_r().round(2)
    
    df = df.drop(['High','Low','Open'],axis=1)

    filtered_df = filter_df(df, end_info_day, day_120)

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
