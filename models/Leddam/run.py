import argparse
import os
import torch
from models.Leddam.exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import numpy as np
import pandas as pd
# def train(ii,args):
#     setting = '{}_pl{}_n_layers_{}_d_model_{}_dropout_{}_pe_type_{}_bs_{}_lr_{}'.format(
#                     args.data_path[:-4],
#                     args.pred_len,
#                     args.n_layers,
#                     args.d_model,
#                     args.dropout,
#                     args.pe_type,
#                     args.batch_size,
#                     args.learning_rate,
#                     )

#     exp = Exp(args)  # set experiments
#     print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
#     exp.train(setting)
#     print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
#     exp.test(setting,test=1)
#     torch.cuda.empty_cache()


#     preds = exp.get_predictions_as_dataframe()

#     print('预测结果：',preds)

    

def get_Leddam_predictions_data(choosecolumn,origin_data, predict_data, **kwargs):



    # # 确保 origin_data 是一个 DataFrame
    # if isinstance(origin_data, list):
    #     origin_data_csv = pd.DataFrame(origin_data).transpose()

    # 确保 origin_data 是一个 DataFrame
    if isinstance(origin_data, list):
        origin_data = pd.DataFrame(origin_data[0])  # 假设 origin_data 是一个三维列表，取第一个元素并转换为 DataFrame
    elif isinstance(origin_data, np.ndarray):
        origin_data = pd.DataFrame(origin_data[0])  # 假设 origin_data 是一个三维数组，取第一个元素并转换为 DataFrame
    
    # 确保 predict_data 是一个 DataFrame
    if isinstance(predict_data, list):
        predict_data = pd.DataFrame(predict_data[0])
    elif isinstance(predict_data, np.ndarray):
        predict_data = pd.DataFrame(predict_data[0])

    # # 确保 origin_data 的索引是日期时间索引
    # if not pd.api.types.is_datetime64_any_dtype(origin_data_csv.index):
    #     origin_data_csv.index = pd.to_datetime(origin_data_csv.index, errors='coerce')
    #     if origin_data_csv.index.isnull().any():
    #         raise ValueError("origin_data 的索引包含无法解析为日期时间格式的值")

    # 把传入的origin_data生成为csv文件，保留索引
    origin_data_path = ('origin_data.csv')

    origin_data.to_csv(origin_data_path, index_label='date')

    # 重新读取 CSV 文件以确保列名称正确
    origin_data_csv = pd.read_csv(origin_data_path, index_col='date', parse_dates=True)

    # 打印 origin_data 的长度
    print('origin_data 的长度:', len(origin_data_csv))
    # 打印要预测的长度
    pred_len = len(predict_data)
    print('要预测的长度:', pred_len)


    # 填充 pred_len 数量的 0
    # 获取当前的索引
    # current_index = origin_data_csv.index


    # pred_len = len(predict_data)

    # # 生成新的索引，扩展 pred_len 个时间步
    # new_index = pd.date_range(start=current_index[0], 
    #                         periods=len(current_index) + pred_len, 
    #                         freq=current_index.freq)

    # # 使用 reindex 方法扩展 DataFrame，并填充新增行的值为 0
    # origin_data_csv = origin_data_csv.reindex(new_index)

    # # 使用线性插值填充新增的行
    # origin_data_csv = origin_data_csv.interpolate(method='linear')

    # #把传入的origin_data生成为csv文件
    # origin_data_csv.to_csv('origin_data.csv', index_label='date')



    

    # # 重新读取 CSV 文件以确保列名称正确
    # origin_data_csv = pd.read_csv(origin_data_path, index_col='date', parse_dates=True)

    

    # #打印origin_data的长度
    # print('origin_data的长度:',len(origin_data_csv))
    # #打印要预测的长度
    # print('要预测的长度:',pred_len)

    

    # 打印填充后的数据长度
    print('填充后的 origin_data 的长度:', len(origin_data_csv))

    print('origin_data的最后200条数据:',origin_data_csv.tail(200))

    #根据传入的orgin_data获取数据的频率
    inferred_freq = pd.infer_freq(origin_data_csv.index)
    print('inferred_freq:',inferred_freq)

    print('origin_data的列:',origin_data_csv.columns)

    #获取预测目标列
    target_col = choosecolumn
    print('target_col:',target_col)

    #根据传入的origin_data获取变量数量
    n_features = origin_data_csv.shape[1]

    print('n_features:',n_features)


    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Leddam')

    # basic config
    
    parser.add_argument('--kernel_size', type=int, default=25, help='kernel_size hyperparameter of smoothing')
    
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='weather', help='model id')
    parser.add_argument('--model', type=str,  default='Leddam')
    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')

#修改文件路径
    parser.add_argument('--root_path', type=str, default='./', help='root path of the data file')
    #parser.add_argument('--data_path', type=str, default='weather.csv', help='data file')
    parser.add_argument('--data_path', type=str, default='origin_data.csv', help='data file')


#选择单多变量预测：M：多变量预测多变量，S：单变量预测单变量，MS：多变量预测单变量
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    
#预测目标列
    #parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    #parser.add_argument('--target', type=str, default='High', help='target feature in S or MS task')
    parser.add_argument('--target', type=str, default=target_col, help='target feature in S or MS task')
    

#预测频率
    #parser.add_argument('--freq', type=str, default='h',
    #                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    #parser.add_argument('--freq', type=str, default='D',
    #                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    
    parser.add_argument('--freq', type=str, default=inferred_freq,
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    
    
    parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='location of model checkpoints')

    # forecasting task
    #分段参数
    #parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of backbone model')
    parser.add_argument('--seq_len', type=int, default=pred_len, help='input sequence length of backbone model')
    

    parser.add_argument('--label_len', type=int, default=pred_len, help='start token length')

#预测长度
    #parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--pred_len', type=int, default=pred_len, help='prediction sequence length')


#修改变量数量
    # model define
    parser.add_argument('--enc_in', type=int, default=n_features, help='encoder input size')
    parser.add_argument('--d_model', type=int, default=256, help='model input size')
    parser.add_argument('--dec_in', type=int, default=n_features, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=n_features, help='output size')


    parser.add_argument('--n_layers', type=int, default=1, help='n_layers of DEFT Block')
    parser.add_argument('--pe_type', type=str, default='no', help='position embedding type')
    parser.add_argument('--dropout', type=float, default=0., help='dropout ratio')
    parser.add_argument('--revin', type=bool, default=True, help='using revin from non-stationary transformer')


    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')

    #训练轮数
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')


    parser.add_argument('--batch_size', type=int, default=24, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=6, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='constant', help='adjust learning rate')
    parser.add_argument('--use_amp', type=bool, default=True, help='use automatic mixed precision training')
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        
    args.label_len=int(args.seq_len//2)
    Exp = Exp_Long_Term_Forecast

    
    
    print('Args in experiment:')
    print(args)
    #train(1,args) 

    setting = '{}_pl{}_n_layers_{}_d_model_{}_dropout_{}_pe_type_{}_bs_{}_lr_{}'.format(
                    args.data_path[:-4],
                    args.pred_len,
                    args.n_layers,
                    args.d_model,
                    args.dropout,
                    args.pe_type,
                    args.batch_size,
                    args.learning_rate,
                    )

    exp = Exp(args)  # set experiments
    print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train(setting)
    print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.test(setting,test=1)
    torch.cuda.empty_cache()


    preds = exp.get_predictions_as_dataframe()



    print('预测结果：',preds)


    out_dict = {
      'NLL/D': np.nan,  # 这里假设没有 NLL/D 的计算
      'samples': preds[target_col].tolist(),
      'median': preds[target_col].tolist(),
      'info': {'Method': 'Leddam', 'model': 'Leddam'}
   }
    
    return out_dict

# if __name__ == '__main__':
#     fix_seed = 2021
#     random.seed(fix_seed)
#     torch.manual_seed(fix_seed)
#     np.random.seed(fix_seed)

#     parser = argparse.ArgumentParser(description='Leddam')

#     # basic config
    
#     parser.add_argument('--kernel_size', type=int, default=25, help='kernel_size hyperparameter of smoothing')
    
#     parser.add_argument('--task_name', type=str, default='long_term_forecast')
#     parser.add_argument('--is_training', type=int, default=1, help='status')
#     parser.add_argument('--model_id', type=str, default='weather', help='model id')
#     parser.add_argument('--model', type=str,  default='Leddam')
#     # data loader
#     parser.add_argument('--data', type=str, default='custom', help='dataset type')

# #修改文件路径
#     parser.add_argument('--root_path', type=str, default='./', help='root path of the data file')
#     #parser.add_argument('--data_path', type=str, default='weather.csv', help='data file')
#     parser.add_argument('--data_path', type=str, default='mul_test.csv', help='data file')


# #选择单多变量预测：M：多变量预测多变量，S：单变量预测单变量，MS：多变量预测单变量
#     parser.add_argument('--features', type=str, default='M',
#                         help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    
#     #预测目标列
#     #parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
#     parser.add_argument('--target', type=str, default='High', help='target feature in S or MS task')
    
#     #预测频率
#     #parser.add_argument('--freq', type=str, default='h',
#     #                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
#     parser.add_argument('--freq', type=str, default='D',
#                         help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    
    
#     parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='location of model checkpoints')

#     # forecasting task
#     #分段参数
#     #parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of backbone model')
#     parser.add_argument('--seq_len', type=int, default=48, help='input sequence length of backbone model')
    

#     parser.add_argument('--label_len', type=int, default=48, help='start token length')


#     #parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
#     parser.add_argument('--pred_len', type=int, default=48, help='prediction sequence length')

# #修改变量数量
#     # model define
#     parser.add_argument('--enc_in', type=int, default=5, help='encoder input size')
#     parser.add_argument('--d_model', type=int, default=256, help='model input size')
#     parser.add_argument('--dec_in', type=int, default=5, help='decoder input size')
#     parser.add_argument('--c_out', type=int, default=5, help='output size')


#     parser.add_argument('--n_layers', type=int, default=1, help='n_layers of DEFT Block')
#     parser.add_argument('--pe_type', type=str, default='no', help='position embedding type')
#     parser.add_argument('--dropout', type=float, default=0., help='dropout ratio')
#     parser.add_argument('--revin', type=bool, default=True, help='using revin from non-stationary transformer')


#     # optimization
#     parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
#     parser.add_argument('--itr', type=int, default=1, help='experiments times')

#     #训练轮数
#     parser.add_argument('--train_epochs', type=int, default=5, help='train epochs')

# #不懂预测长度和 batch_size的具体关系
# #预测长度是batch_size-2
#     parser.add_argument('--batch_size', type=int, default=18, help='batch size of train input data')
#     parser.add_argument('--patience', type=int, default=6, help='early stopping patience')
#     parser.add_argument('--learning_rate', type=float, default=5e-4, help='optimizer learning rate')
#     parser.add_argument('--des', type=str, default='Exp', help='exp description')
#     parser.add_argument('--loss', type=str, default='mse', help='loss function')
#     parser.add_argument('--lradj', type=str, default='constant', help='adjust learning rate')
#     parser.add_argument('--use_amp', type=bool, default=True, help='use automatic mixed precision training')
#     # GPU
#     parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
#     parser.add_argument('--gpu', type=int, default=0, help='gpu')
#     parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
#     parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')


#     args = parser.parse_args()
#     args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
#     if args.use_gpu and args.use_multi_gpu:
#         args.devices = args.devices.replace(' ', '')
#         device_ids = args.devices.split(',')
#         args.device_ids = [int(id_) for id_ in device_ids]
#         args.gpu = args.device_ids[0]
        
#     args.label_len=int(args.seq_len//2)
#     Exp = Exp_Long_Term_Forecast

    
    
#     print('Args in experiment:')
#     print(args)
#     train(1,args) 
    
    
            
                      
