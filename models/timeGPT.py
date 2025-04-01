from nixtla import NixtlaClient

import pandas as pd
import numpy as np
import plotly.graph_objects as go


from nixtla import NixtlaClient
import pandas as pd


def get_timeGPT_predictions_data(choosecolumn, origin_data, predict_data, **kwargs):
    """
    获取 timeGPT 模型的预测数据，支持多变量数据
    :param choosecolumn: 选择的列名（目标列）
    :param origin_data: 原始数据，传入的是 DataFrame 或包含 DataFrame 的列表
    :param predict_data: 预测数据，传入的是 DataFrame 或包含 DataFrame 的列表
    :param kwargs: 其他参数
    :return: 预测结果的字典
    """
    # 确保 origin_data 是 DataFrame
    if isinstance(origin_data, list):
        origin_data = origin_data[0]  # 假设传入的是包含单个 DataFrame 的列表
    if not isinstance(origin_data, pd.DataFrame):
        raise ValueError("origin_data 必须是一个 pandas.DataFrame 或包含单个 DataFrame 的列表")

    # 确保 predict_data 是 DataFrame
    if isinstance(predict_data, list):
        predict_data = predict_data[0]  # 假设传入的是包含单个 DataFrame 的列表
    if not isinstance(predict_data, pd.DataFrame):
        raise ValueError("predict_data 必须是一个 pandas.DataFrame 或包含单个 DataFrame 的列表")


    # 确保时间列是第一列
    if not pd.api.types.is_datetime64_any_dtype(origin_data.index):
        raise ValueError("origin_data 的索引必须是日期时间类型")
    if not pd.api.types.is_datetime64_any_dtype(predict_data.index):
        raise ValueError("predict_data 的索引必须是日期时间类型")

   # 推断频率
    freq = pd.infer_freq(origin_data.index)
    if freq is None:
        # 如果无法推断频率，尝试手动设置频率
        freq = kwargs.get('freq', 'D')  # 默认使用 'D'（每天）
        print(f"无法推断频率，使用手动设置的频率: {freq}")

    print('提取的频率:', freq)

    # 将索引转换为列，并重置索引
    origin_df = origin_data.reset_index()
    predict_df = predict_data.reset_index()

    # 打印提取到的列名
    print('origin_df 的列名:', origin_df.columns)
    print('predict_df 的列名:', predict_df.columns)

    # 获取时间列和目标列
    time_col = origin_df.columns[0]  # 时间列名
    target_col = choosecolumn  # 目标列名由参数传入

    # 检查目标列是否存在
    if target_col not in origin_df.columns:
        raise ValueError(f"目标列 '{target_col}' 不在 origin_data 中")

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
    if 'TimeGPT' in timegpt_fcst_df.columns:
        target_col = 'TimeGPT'

    # 重命名预测结果的列名为目标列名
    timegpt_fcst_df.rename(columns={'value': target_col}, inplace=True)

    # 构建输出字典
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


# def get_timeGPT_predictions_data(choosecolumn,origin_data, predict_data, **kwargs):
#    """
#    获取 timeGPT 模型的预测数据
#    :param origin_data: 原始数据，传入的是 list 类型，包含时间戳和目标值
#    :param predict_data: 预测数据，传入的是 list 类型，包含时间戳和占位值
#    :param kwargs: 其他参数
#    :return: 预测结果的字典
#    """
#    #从传入的origin_data中提取Freq参数
#    #提取 Series


#    series = origin_data[0]

#    # 获取频率
#    freq = series.index.freq

#    print('提取的频率:', freq)

#    # 将 origin_data 转换为 DataFrame
#    if isinstance(origin_data, list) and len(origin_data) == 1:
#       origin_series = origin_data[0]
#       if isinstance(origin_series, pd.Series):
#          # 将 Series 转换为 DataFrame，并将索引转换为列
#          origin_df = origin_series.reset_index()
#          #origin_df.columns = ['ds', 'y']  # 假设时间列名为 'ds'，目标列名为 'y'
#          # 使用 Series 的名称作为列名
#          origin_df.columns = ['ds', origin_series.name]  # 时间列名为 'ds'，目标列名为 Series 的名称
#       else:
#          raise ValueError("origin_data 中的元素必须是 pandas.Series")
#    else:
#       raise ValueError("origin_data 必须是一个包含单个 pandas.Series 的列表")

#    # 将 predict_data 转换为 DataFrame
#    if isinstance(predict_data, list) and len(predict_data) == 1:
#       predict_series = predict_data[0]
#       if isinstance(predict_series, pd.Series):
#          # 将 Series 转换为 DataFrame，并将索引转换为列
#          predict_df = predict_series.reset_index()
#          #predict_df.columns = ['ds', 'y']  # 假设时间列名为 'ds'，目标列名为 'y'
#          # 使用 Series 的名称作为列名
#          predict_df.columns = ['ds', predict_series.name]  # 时间列名为 'ds'，目标列名为 Series 的名称
#       else:
#          raise ValueError("predict_data 中的元素必须是 pandas.Series")
#    else:
#       raise ValueError("predict_data 必须是一个包含单个 pandas.Series 的列表")

#    #打印提取到的列名
#    print('origin_df的列名:', origin_df.columns)
#    print('predict_df的列名:', predict_df.columns)

#    # 获取列名称
#    time_col = origin_df.columns[0]  # 时间列名
#    target_col = origin_df.columns[1]  # 目标列名

#    # 初始化 NixtlaClient
#    nixtla_client = NixtlaClient(api_key='nixak-Dwlb7vJzTNvbKpGxYdJDwco0shomWEXfjaXcDaC1UkHi0Ndvfqv3gholUZPhFgaw0zVUD9BnubnLbkpN')

#    # 获取预测结果
#    timegpt_fcst_df = nixtla_client.forecast(
#       df=origin_df, 
#       h=len(predict_df), 
#       freq=freq,
#       time_col=time_col, 
#       target_col=target_col, 
#       model='timegpt-1-long-horizon'
#    )

#    # 打印预测结果的列名，以确认正确的列名
#    print('timeGPT 预测结果的列名:', timegpt_fcst_df.columns)

#    # 动态设置 target_col
#    # 假设预测结果的列名是 'value' 或 'forecast'
#    # 使用提取到的origin_df中的列名作为预测结果的列名

#    if 'TimeGPT' in timegpt_fcst_df.columns:
#         target_col = 'TimeGPT'

#    # 重命名预测结果的列名为目标列名
#    timegpt_fcst_df.rename(columns={'value': target_col}, inplace=True)



# #缺少NLL/D的计算
#    out_dict = {
#       'NLL/D': np.nan,  # 这里假设没有 NLL/D 的计算
#       'samples': timegpt_fcst_df[target_col].tolist(),
#       'median': timegpt_fcst_df[target_col].tolist(),
#       'info': {'Method': 'timeGPT', 'model': 'timegpt-1-long-horizon'}
#    }

#    # 打印预测结果结构
#    print('timeGPT 预测结果:', out_dict)
#    print('timeGPT 预测结果结构:', out_dict.keys())

#    return out_dict




