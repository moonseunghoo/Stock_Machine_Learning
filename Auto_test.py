import smtplib
import pandas as pd
import cloudscraper
import time
import FinanceDataReader as fdr
import tensorflow as tf

#tf.config.experimental.set_visible_devices([], 'GPU')

from collections import OrderedDict
from datetime import datetime, timedelta
from defs_pred_auto import Data_Scrap_Pred
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from io import BytesIO

#e-mail 함수
def send_email(subject, body):
    smtp_server = "smtp.gmail.com"
    port = 587  # Gmail SMTP 포트

    sender_email = "hoo217606@gmail.com"
    receiver_email = "tmdgn2002@gmail.com"
    app_password = "yoch spra idlc stki"  # 애플리케이션 비밀번호 사용

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject  # 수정된 부분

    message.attach(MIMEText(body, "plain"))

    server = smtplib.SMTP(smtp_server, port)
    server.starttls()
    server.login(sender_email, app_password)
    server.sendmail(sender_email, receiver_email, message.as_string())
    server.quit()

#krx정보데이터 크롤링
def KRX_Crolling(otp_form_data):
  # generate.cmd 요청 주소 
  otp_url ='http://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd' 
  # download.cmd 요청 주소
  csv_url = 'http://data.krx.co.kr/comm/fileDn/download_csv/download.cmd'

  # scraper 객체 
  scraper = cloudscraper.create_scraper()

  # form data와 함께 요청 
  # response의 text 에 담겨있는 otp 코드 가져오기
  otp = scraper.post(otp_url, params=otp_form_data).text

  date = otp_form_data['trdDd'][2:]

  csv_form_data = scraper.post(csv_url, params={'code': otp})
  stock_df = pd.read_csv(BytesIO(csv_form_data.content), encoding='euc-kr')

  if otp_form_data['money'] == '3':
    stock_df.to_csv('KRX/kospi/' + date + '.csv',index=False, encoding='euc-kr')
  else :
    stock_df.to_csv('KRX/marcap/' + date + '.csv',index=False, encoding='euc-kr')

  return 

# 현재 날짜를 가져오는 함수
def get_current_date():
    return datetime.now().strftime('%Y%m%d')

# krx정보데이터 크롤링에 필요한 데이터 
def KRX_data_form():
  # form data 
  kospi_form_data = {
    'locale': 'ko_KR',
    'idxIndMidclssCd': '02',
    'trdDd': '20240308',
    "share": '2',
    'money': '3',
    "csvxls_isNo": 'false',
    "name": 'fileDown',
    "url": 'dbms/MDC/STAT/standard/MDCSTAT00101'
  }
  kospi_form_data['trdDd'] = get_current_date()

  marcap_form_data = {
    'locale': 'ko_KR',
    'mktId': 'ALL',
    'trdDd': '20240308',
    'share': '1',
    'money': '1',
    'csvxls_isNo': 'false',
    'name': 'fileDown',
    'url': 'dbms/MDC/STAT/standard/MDCSTAT01501'
  }
  marcap_form_data['trdDd'] = get_current_date()

  return kospi_form_data, marcap_form_data

def make_six_digit_list(input_list):
  six_digit_list = []

  for item in input_list:
      # 현재 항목의 길이를 확인
      item_str = str(item)
      item_len = len(item_str)

      if item_len < 6:
          # 6자리가 안되는 항목은 앞에 0을 채워서 6자리로 만듭니다.
          zero_padding = '0' * (6 - item_len)
          six_digit_item = zero_padding + item_str
          six_digit_list.append(six_digit_item)
      else:
          # 이미 6자리인 경우 그대로 유지
          six_digit_list.append(item_str)

  return six_digit_list

def filter_df(df): #데이터프레임 필터링
  # 첫 번째 열에서 같은 값을 가진 행의 수를 계산합니다.
  row_counts = df['Ticker'].value_counts()

  # 가장 많은 행의 수를 찾습니다.
  max_row_count = row_counts.max()

  # 가장 많은 행의 수에 해당하는 행만 분류합니다.
  filtered = pd.DataFrame(df[df['Ticker'].isin(row_counts[row_counts == max_row_count].index)])

  return filtered

def Prediction():
  # today = datetime.now()
  today = (datetime.now() - timedelta(days=3))
  if today.strftime('%a') == 'Fri':
      target_day = (today + timedelta(days=3))
  else :
      target_day = (today + timedelta(days=1))

  if target_day.strftime('%a') == 'Mon':
      end_info_day = (target_day - timedelta(days=3))
  else :
      end_info_day = (target_day - timedelta(days=1))

  date = target_day.strftime("%Y-%m-%d").replace('-', '')[2:]
  print('예측에 필요한 파일 날짜 : ', date)

  pred_df = pd.read_csv('KRX/Scrap_Pred/StockData_Pred_'+ date +'.csv',low_memory=False)

  filter_pred= filter_df(pred_df)

  # 예측에 필요한 데이터 
  pred_ticker = list(OrderedDict.fromkeys(filter_pred['Ticker'])) #종목코드 저장

  # 불필요한 데이터 삭제
  filter_pred = filter_pred.drop({'Ticker','Date','Change'},axis=1) #종목코드, 날자, 상승율 삭제

  model = �tf.keras.models.load_model("RaspberryPi_test.h5")
  
  # GRU_128_64_32_2_KOSPI_TI_3%.h5
  Pred = model.predict(filter_pred).round(2)
  # 5% 이상 오를 종목 식별
  rising_stocks = [ticker for i, ticker in enumerate(pred_ticker) if Pred[i] > 0.9]
  rising_stocks = list(map(str, rising_stocks))
  result = list(set(rising_stocks))
  result = make_six_digit_list(result)
  high_52 = add_52_week_high_info(result,end_info_day)
  final_result = high_52.head(5)
  return final_result

def add_52_week_high_info(series,end_info_day):
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

      # 거래량(거래대금)의 평균 계산
      stock_data = stock_data.tail(5)
      average_volume = int(stock_data['Volume'].mean())
      
      high_52_week_list.append(rolling_result.tail(1))
      change_percentage_list.append(change_percentage.tail(1))
      average_volume_5_days.append(average_volume)

  df['52주 최고가'] = high_52_week_list
  df['52주 최고가 대비 변동율'] = change_percentage_list
  df['5일 평균 거래대금'] = average_volume_5_days
  #5일 평균 거래량 기준 내림차순
  df = df.sort_values(by='5일 평균 거래대금', ascending=False)

  return df

if __name__ == '__main__':
  # today = datetime.now()
  today = (datetime.now() - timedelta(days=3))

  if today.strftime('%a') == 'Fri':
      targer_day = (today + timedelta(days=3)).strftime('%Y%m%d')[2:]
  else :
      targer_day = (today + timedelta(days=1)).strftime('%Y%m%d')[2:]
  print('예측에 필요한 파일 저장 날짜 : ', targer_day)

  # kospi_form_data, marcap_form_data = KRX_data_form()
  # KRX_Crolling(kospi_form_data)
  # KRX_Crolling(marcap_form_data)
  
  # s_df = Data_Scrap_Pred()
  # s_df.to_csv('KRX/Scrap_Pred/StockData_Pred_'+ targer_day +'.csv',index=False)

  pred = Prediction().to_string(index=False)
  print(pred)
  # 이메일 보내기
  # send_email(targer_day + ' 3%이상 상승 예측 종목', pred)
  
