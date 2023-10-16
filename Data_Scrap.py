import multiprocessing
import pandas as pd
import defs
import time
from pykrx import stock
from tqdm import tqdm

def Data_Scrap():
    code_list = stock.get_market_ticker_list('230831', market='KOSPI')
    kosdaq = stock.get_market_ticker_list('20231011',market='KOSDAQ')
    code_list = code_list + kosdaq
    # code_list = ['005930','030200']

    stock_data = []
    ## 멀티 프로세싱 ###
    p = multiprocessing.Pool(processes=6)
    for row in tqdm(p.map(defs.merging_stock_data,code_list),mininterval=1):
        stock_data += row
    p.close()
    p.join()
    s_df = pd.DataFrame(stock_data)
    #timestamp 형식 int로 변환
    s_df[1] = s_df[1].dt.year * 10000 + s_df[1].dt.month * 100 + s_df[1].dt.day
    s_df = s_df[~s_df[0].str.contains('K|L|M')]

    return s_df


if __name__ == '__main__':
    start_time = time.time()
    s_df = Data_Scrap()
    s_df.to_csv('/Users/moon/Desktop/Moon SeungHoo/Stock_Machine_Learning/StockData.csv',index=False)
    endtime = time.time()
    print('---걸린시간 : {} ---',format((endtime - start_time)/6))
    
