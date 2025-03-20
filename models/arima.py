from statsmodels.tsa.arima.model import ARIMA as staARIMA
import numpy as np
import pandas as pd
from darts import TimeSeries
import darts.models
from sklearn.preprocessing import MinMaxScaler


def _new_arima_fit(self, series, future_covariates=None):
    """
    重写ARIMA模型的拟合函数，支持单变量时间序列的拟合。
    
    参数:
    - series: 时间序列数据（单变量）。
    - future_covariates: 未来协变量（可选）。
    
    返回:
    - self: 拟合后的模型对象。
    """
    # 调用父类的拟合方法
    super(darts.models.ARIMA, self)._fit(series, future_covariates)

    # 确保时间序列是单变量的
    self._assert_univariate(series)

    # 存储未来协变量，以便后续使用
    self.training_historic_future_covariates = future_covariates

    # 使用statsmodels的ARIMA模型进行拟合
    m = staARIMA(
        series.values(copy=False),  # 时间序列数据
        exog=future_covariates.values(copy=False) if future_covariates else None,  # 未来协变量
        order=self.order,  # ARIMA模型的(p, d, q)参数
        seasonal_order=self.seasonal_order,  # 季节性ARIMA模型的参数
        trend=self.trend,  # 趋势项
    )
    # 拟合模型并存储在self.model中
    self.model = m.fit()

    return self


# def get_arima_predictions_data(train, test, p=12, d=1, q=0, num_samples=100, **kwargs):
#     num_samples = max(num_samples, 1)
#     if not isinstance(train, list):
#         assume single train/test case
#         train = [train]
#         test = [test]
#     for i in range(len(train)):    
#         if not isinstance(train[i], pd.Series):
#             train[i] = pd.Series(train[i], index = pd.RangeIndex(len(train[i])))
#             test[i] = pd.Series(test[i], index = pd.RangeIndex(len(train[i]),len(test[i])+len(train[i])))

#     test_len = len(test[0])
#     assert all(len(t)==test_len for t in test), f'All test series must have same length, got {[len(t) for t in test]}'

#     model = darts.models.ARIMA(p=p, d=d, q=q)

#     scaled_train_ts_list = []
#     scaled_test_ts_list = []
#     scaled_combined_series_list = []
#     scalers = []


#     Iterate over each series in the train list
#     for train_series, test_series in zip(train,test):
#         for ARIMA we scale each series individually
#         scaler = MinMaxScaler()
#         combined_series = pd.concat([train_series,test_series])
#         scaler.fit(combined_series.values.reshape(-1,1))
#         scalers.append(scaler)
#         scaled_train_series = scaler.transform(train_series.values.reshape(-1,1)).reshape(-1)
#         scaled_train_series_ts = TimeSeries.from_times_and_values(train_series.index, scaled_train_series)
#         scaled_train_ts_list.append(scaled_train_series_ts)

#         scaled_test_series = scaler.transform(test_series.values.reshape(-1,1)).reshape(-1)
#         scaled_test_series_ts = TimeSeries.from_times_and_values(test_series.index, scaled_test_series)
#         scaled_test_ts_list.append(scaled_test_series_ts)
        
#         scaled_combined_series = scaler.transform(pd.concat([train_series,test_series]).values.reshape(-1,1)).reshape(-1)
#         scaled_combined_series_list.append(scaled_combined_series)
        

#     rescaled_predictions_list = []
#     nll_all_list = []
#     samples_list = []

#     for i in range(len(scaled_train_ts_list)):
#         try:
#             model.fit(scaled_train_ts_list[i])
#             prediction = model.predict(len(test[i]), num_samples=num_samples).data_array()[:,0,:].T.values
#             scaler = scalers[i]
#             rescaled_prediction = scaler.inverse_transform(prediction.reshape(-1,1)).reshape(num_samples,-1)
#             fit_model = model.model.model.fit()
#             fit_params = fit_model.conf_int().mean(1)
#             all_model = staARIMA(
#                     scaled_combined_series_list[i],
#                     exog=None,
#                     order=model.order,
#                     seasonal_order=model.seasonal_order,
#                     trend=model.trend,
#             )
#             nll_all = -all_model.loglikeobs(fit_params)
#             nll_all = nll_all[len(train[i]):].sum()/len(test[i])
#             nll_all -= np.log(scaler.scale_)
#             nll_all = nll_all.item()
#         except np.linalg.LinAlgError:
#             rescaled_prediction = np.zeros((num_samples,len(test[i])))
#             output nan
#             nll_all = np.nan

#         samples = pd.DataFrame(rescaled_prediction, columns=test[i].index)
        
#         rescaled_predictions_list.append(rescaled_prediction)
#         nll_all_list.append(nll_all)
#         samples_list.append(samples)
        
#     out_dict = {
#         'NLL/D': np.mean(nll_all_list),
#         'samples': samples_list if len(samples_list)>1 else samples_list[0],
#         'median': [samples.median(axis=0) for samples in samples_list] if len(samples_list)>1 else samples_list[0].median(axis=0),
#         'info': {'Method':'ARIMA', 'p':p, 'd':d}
#     }

#     return out_dict

def get_arima_predictions_data(train, test, p=12, d=1, q=0, num_samples=100, **kwargs):
    """
    获取ARIMA模型的预测数据，支持多组训练和测试数据。
    
    参数:
    - train: 训练数据（可以是单个Series或列表）。
    - test: 测试数据（可以是单个Series或列表）。
    - p: ARIMA模型的自回归阶数。
    - d: ARIMA模型的差分阶数。
    - q: ARIMA模型的移动平均阶数。
    - num_samples: 生成的预测样本数量。
    
    返回:
    - out_dict: 包含预测结果、中位数、负对数似然值等信息的字典。
    """
    # 确保num_samples至少为1
    num_samples = max(num_samples, 1)

    # 如果train和test不是列表，将其转换为列表形式
    if not isinstance(train, list):
        train = [train]
        test = [test]

    # 将train和test中的元素转换为pandas.Series，并为其创建索引
    for i in range(len(train)):
        if not isinstance(train[i], pd.Series):
            train[i] = pd.Series(train[i], index=pd.RangeIndex(len(train[i])))
            test[i] = pd.Series(test[i], index=pd.RangeIndex(len(train[i]), len(test[i]) + len(train[i])))

    # 确保所有测试数据的长度相同
    test_len = len(test[0])
    assert all(len(t) == test_len for t in test), f'所有测试序列的长度必须相同，当前长度: {[len(t) for t in test]}'

    # 初始化ARIMA模型
    model = darts.models.ARIMA(p=p, d=d, q=q)

    # 初始化列表，用于存储缩放后的训练数据、测试数据和组合数据，以及缩放器
    scaled_train_ts_list = []
    scaled_test_ts_list = []
    scaled_combined_series_list = []
    scalers = []

    # 对每个训练和测试数据进行归一化处理，并将其转换为TimeSeries对象
    for train_series, test_series in zip(train, test):
        scaler = MinMaxScaler()
        combined_series = pd.concat([train_series, test_series])
        scaler.fit(combined_series.values.reshape(-1, 1))  # 拟合缩放器
        scalers.append(scaler)

        # 缩放训练数据
        scaled_train_series = scaler.transform(train_series.values.reshape(-1, 1)).reshape(-1)
        scaled_train_series_ts = TimeSeries.from_times_and_values(train_series.index, scaled_train_series)
        scaled_train_ts_list.append(scaled_train_series_ts)

        # 缩放测试数据
        scaled_test_series = scaler.transform(test_series.values.reshape(-1, 1)).reshape(-1)
        scaled_test_series_ts = TimeSeries.from_times_and_values(test_series.index, scaled_test_series)
        scaled_test_ts_list.append(scaled_test_series_ts)

        # 缩放组合数据
        scaled_combined_series = scaler.transform(pd.concat([train_series, test_series]).values.reshape(-1, 1)).reshape(-1)
        scaled_combined_series_list.append(scaled_combined_series)

    # 初始化列表，用于存储重新缩放后的预测结果、负对数似然值和样本
    rescaled_predictions_list = []
    nll_all_list = []
    samples_list = []

    # 对每个归一化后的训练数据进行模型拟合，并生成预测结果
    for i in range(len(scaled_train_ts_list)):
        try:
            # 拟合模型
            model.fit(scaled_train_ts_list[i])
            # 生成预测结果
            prediction = model.predict(len(test[i]), num_samples=num_samples).data_array()[:, 0, :].T.values
            scaler = scalers[i]
            # 将预测结果重新缩放回原始尺度
            rescaled_prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).reshape(num_samples, -1)

            # 计算负对数似然值（NLL），用于评估模型性能
            fit_model = model.model.model.fit()
            fit_params = fit_model.conf_int().mean(1)
            all_model = staARIMA(
                scaled_combined_series_list[i],
                exog=None,
                order=model.order,
                seasonal_order=model.seasonal_order,
                trend=model.trend,
            )
            nll_all = -all_model.loglikeobs(fit_params)
            nll_all = nll_all[len(train[i]):].sum() / len(test[i])
            nll_all -= np.log(scaler.scale_)
            nll_all = nll_all.item()
        except np.linalg.LinAlgError:
            # 如果出现线性代数错误，则输出零矩阵和NaN
            rescaled_prediction = np.zeros((num_samples, len(test[i])))
            nll_all = np.nan

        # 将预测结果存储在DataFrame中
        samples = pd.DataFrame(rescaled_prediction, columns=test[i].index)
        rescaled_predictions_list.append(rescaled_prediction)
        nll_all_list.append(nll_all)
        samples_list.append(samples)

    # 将结果存储在字典中
    out_dict = {
        'NLL/D': np.mean(nll_all_list),  # 平均负对数似然值
        'samples': samples_list if len(samples_list) > 1 else samples_list[0],  # 预测样本
        'median': [samples.median(axis=0) for samples in samples_list] if len(samples_list) > 1 else samples_list[0].median(axis=0),  # 预测中位数
        'info': {'Method': 'ARIMA', 'p': p, 'd': d}  # 模型信息
    }

    return out_dict