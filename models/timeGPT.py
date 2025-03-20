from nixtla import NixtlaClient

import pandas as pd
import numpy as np
import plotly.graph_objects as go


from nixtla import NixtlaClient
import pandas as pd



def get_timeGPT_predictions_data(origin_data, predict_data, **kwargs):
   """
   获取 timeGPT 模型的预测数据
   :param origin_data: 原始数据，传入的是 list 类型，包含时间戳和目标值
   :param predict_data: 预测数据，传入的是 list 类型，包含时间戳和占位值
   :param kwargs: 其他参数
   :return: 预测结果的字典
   """
   #从传入的origin_data中提取Freq参数
   #提取 Series
   series = origin_data[0]

   # 获取频率
   freq = series.index.freq

   print('提取的频率:', freq)

   # 将 origin_data 转换为 DataFrame
   if isinstance(origin_data, list) and len(origin_data) == 1:
      origin_series = origin_data[0]
      if isinstance(origin_series, pd.Series):
         # 将 Series 转换为 DataFrame，并将索引转换为列
         origin_df = origin_series.reset_index()
         #origin_df.columns = ['ds', 'y']  # 假设时间列名为 'ds'，目标列名为 'y'
         # 使用 Series 的名称作为列名
         origin_df.columns = ['ds', origin_series.name]  # 时间列名为 'ds'，目标列名为 Series 的名称
      else:
         raise ValueError("origin_data 中的元素必须是 pandas.Series")
   else:
      raise ValueError("origin_data 必须是一个包含单个 pandas.Series 的列表")

   # 将 predict_data 转换为 DataFrame
   if isinstance(predict_data, list) and len(predict_data) == 1:
      predict_series = predict_data[0]
      if isinstance(predict_series, pd.Series):
         # 将 Series 转换为 DataFrame，并将索引转换为列
         predict_df = predict_series.reset_index()
         #predict_df.columns = ['ds', 'y']  # 假设时间列名为 'ds'，目标列名为 'y'
         # 使用 Series 的名称作为列名
         predict_df.columns = ['ds', predict_series.name]  # 时间列名为 'ds'，目标列名为 Series 的名称
      else:
         raise ValueError("predict_data 中的元素必须是 pandas.Series")
   else:
      raise ValueError("predict_data 必须是一个包含单个 pandas.Series 的列表")

   #打印提取到的列名
   print('origin_df的列名:', origin_df.columns)
   print('predict_df的列名:', predict_df.columns)

   # 获取列名称
   time_col = origin_df.columns[0]  # 时间列名
   target_col = origin_df.columns[1]  # 目标列名

   # 初始化 NixtlaClient
   nixtla_client = NixtlaClient(api_key='nixak-Dwlb7vJzTNvbKpGxYdJDwco0shomWEXfjaXcDaC1UkHi0Ndvfqv3gholUZPhFgaw0zVUD9BnubnLbkpN')

   # 获取预测结果
   timegpt_fcst_df = nixtla_client.forecast(
      df=origin_df, 
      h=len(predict_df), 
      freq=freq,
      time_col=time_col, 
      target_col=target_col, 
      model='timegpt-1-long-horizon'
   )

   # 打印预测结果的列名，以确认正确的列名
   print('timeGPT 预测结果的列名:', timegpt_fcst_df.columns)

   # 动态设置 target_col
   # 假设预测结果的列名是 'value' 或 'forecast'
   # 使用提取到的origin_df中的列名作为预测结果的列名

   if 'TimeGPT' in timegpt_fcst_df.columns:
        target_col = 'TimeGPT'

   # 重命名预测结果的列名为目标列名
   timegpt_fcst_df.rename(columns={'value': target_col}, inplace=True)



   # # 返回预测结果
   # result = {
   #    'median': timegpt_fcst_df[target_col]  # 假设预测结果在目标值列中
   # }

   # # 打印预测结果结构
   # print('timeGPT 预测结果:', result)
   # print('timeGPT 预测结果结构:', result.keys())

   # return result
   # 包装返回结果

#缺少NLL/D的计算
   out_dict = {
      'NLL/D': np.nan,  # 这里假设没有 NLL/D 的计算
      'samples': timegpt_fcst_df[target_col].tolist(),
      'median': timegpt_fcst_df[target_col].tolist(),
      'info': {'Method': 'timeGPT', 'model': 'timegpt-1-long-horizon'}
   }

   # 打印预测结果结构
   print('timeGPT 预测结果:', out_dict)
   print('timeGPT 预测结果结构:', out_dict.keys())

   return out_dict






# def get_timeGPT_predictions_data(origin_data, predict_data, **kwargs):
#    """
#    获取 timeGPT 模型的预测数据
#    :param origin_data: 原始数据
#    :param predict_data: 预测数据
#    :param kwargs: 其他参数
#    :return: 预测结果的字典
#    """

#    #打印传入的数据
#    print('传入origin_data',origin_data)
#    print(type(origin_data))
#    print('传入predict_data',predict_data)
#    print(type(predict_data))

#    # 将 origin_data 和 predict_data 转换为 DataFrame
#    if isinstance(origin_data, pd.Series):
#       origin_data = origin_data.to_frame()
#    if isinstance(predict_data, pd.Series):
#       predict_data = predict_data.to_frame()

   
#    # 获取列名称
#    time_col = origin_data.columns[0]
#    target_col = origin_data.columns[1]


# # 自动推断频率
#    inferred_freq = pd.infer_freq(origin_data.index)
#    if inferred_freq is None:
#       raise ValueError("无法推断时间序列的频率，请确保时间戳是规则的")



#    # 确保时间戳是规则的
#    origin_data[time_col] = pd.to_datetime(origin_data[time_col])
#    origin_data = origin_data.set_index(time_col).asfreq(inferred_freq)  # 根据需要调整频率


# #考虑后续freq参数如何传送进来




#    # 提供 X_df
#    X_df = origin_data[[target_col]]  # 根据需要选择列

# #多变量时其他的变量设为外生变量

#    # 初始化 NixtlaClient
#    nixtla_client = NixtlaClient(api_key='nixak-Dwlb7vJzTNvbKpGxYdJDwco0shomWEXfjaXcDaC1UkHi0Ndvfqv3gholUZPhFgaw0zVUD9BnubnLbkpN')

#    # 获取预测结果
#    timegpt_fcst_df = nixtla_client.forecast(df=X_df, h=len(predict_data), time_col=time_col, target_col=target_col, model='timegpt-1-long-horizon')

#    # 返回预测结果
#    result = {
#       'median': timegpt_fcst_df[target_col]  # 假设预测结果在 'value' 列中
#    }

#    #打印预测结果结构
#    print('timeGPT预测结果',result)
#    print('timeGPT预测结果结构',result.keys())


#    return result

# df = pd.read_csv('./data/mul_test.csv')
# df.head()


# # CSV 文件的第一列是时间戳，其他列是变量
# #df.rename(columns={df.columns[0]: 'ds', df.columns[1]: 'y'}, inplace=True)

# #确保时间戳是规则的
# df['Date'] = pd.to_datetime(df['Date'])
# df = df.set_index('Date').asfreq('D').reset_index()  # 将频率设置为每日


# # 处理缺失值
# df['Volume'].fillna(method='pad', inplace=True)


# 提供 X_df
#X_df = df['Open', 'High', 'Low', 'Close', 'Adj Close']

# nixtla_clinet = NixtlaClient(# defaults to os.environ.get("TIMEGPT_TOKEN")
#        api_key ='nixak-Dwlb7vJzTNvbKpGxYdJDwco0shomWEXfjaXcDaC1UkHi0Ndvfqv3gholUZPhFgaw0zVUD9BnubnLbkpN')


# timegpt_fcst_df = nixtla_clinet.forecast(df=df, h=7, time_col='Date', target_col='Volume', model='timegpt-1-long-horizon')
# timegpt_fcst_df.head()

# fig = nixtla_clinet.plot(df, timegpt_fcst_df, time_col='Date', target_col='Volume')

# fig.show()



# def get_timegpt_predictions_data(df, h, freq,time_col, target_col):
    
    
    
#     return timegpt_fcst_df

