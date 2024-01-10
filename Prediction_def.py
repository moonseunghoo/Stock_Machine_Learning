import tensorflow as tf
import pandas as pd
from collections import OrderedDict


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
    row_counts = df['0'].value_counts()

    # 가장 많은 행의 수를 찾습니다.
    max_row_count = row_counts.max()

    # 가장 많은 행의 수에 해당하는 행만 분류합니다.
    filtered = pd.DataFrame(df[df['0'].isin(row_counts[row_counts == max_row_count].index)])

    return filtered

def Prediction():
    # 데이터 로드   
    pred_df = pd.read_csv('/Users/moon/Desktop/Moon SeungHoo/Stock_Machine_Learning/StockData_Pred_240110.csv',low_memory=False)

    filter_pred= filter_df(pred_df)

    # 예측에 필요한 데이터 
    pred_ticker = list(OrderedDict.fromkeys(filter_pred['0'])) #종목코드 저장

    # 불필요한 데이터 삭제
    filter_pred = filter_pred.drop({'0','1','7'},axis=1) #종목코드, 날자, 상승율 삭제
    
    model_1 = tf.keras.models.load_model("GRU_Model_7L_15_5%.h5")
    Pred = model_1.predict(filter_pred)

    # 5% 이상 오를 종목 식별
    rising_stocks = [ticker for i, ticker in enumerate(pred_ticker) if Pred[i] > 0.9]

    # T_pred를 기준으로 내림차순 정렬
    rising_stocks_sorted = sorted(rising_stocks, key=lambda i: Pred[pred_ticker.index(i)], reverse=True)
    rising_stocks_sorted = list(map(str, rising_stocks_sorted))
    result = list(set(rising_stocks_sorted))
    result = make_six_digit_list(result)
    sorted_list = result[:5]
    print("내일 5% 이상 상승할 종목:", sorted_list)

    return sorted_list

if __name__ == '__main__':
    make_six_digit_list()
    Prediction()
    filter_df()
