from tqdm import tqdm
from forecast.utils import serialize_arr, deserialize_str, SerializerSettings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

from dataclasses import dataclass
from models.llms import completion_fns, nll_fns, tokenization_fns, context_lengths


# 步长乘数，用于调整预测步长
STEP_MULTIPLIER = 1.2

@dataclass
class Scaler:
    """
    数据缩放器类，用于对数据进行缩放和逆缩放。

    属性:
    - transform (callable): 数据缩放函数。
    - inv_transform (callable): 数据逆缩放函数。
    """
    transform: callable = lambda x: x
    inv_transform: callable = lambda x: x    

#根据历史数据生成一个缩放器对象
def get_scaler(history, alpha=0.95, beta=0.3, basic=False):
    """

    参数:
    - history (array-like): 用于生成缩放器的历史数据。
    - alpha (float, optional): 用于计算缩放的分位数，默认为0.95。
    - beta (float, optional): 偏移参数，默认为0.3。
    - basic (bool, optional): 如果为True，则不应用偏移，并避免对小于0.01的值进行缩放，默认为False。

    返回:
    - Scaler: 配置好的缩放器对象。
    """    
    # 去除历史数据中的NaN值
    history = history[~np.isnan(history)]
    if basic:
        # 基础缩放：仅使用分位数进行缩放
        q = np.maximum(np.quantile(np.abs(history), alpha), .01)
        def transform(x):
            return x / q
        def inv_transform(x):
            return x * q
    else:
        # 高级缩放：考虑偏移和分位数
        min_ = np.min(history) - beta * (np.max(history) - np.min(history))
        q = np.quantile(history - min_, alpha)
        if q == 0:
            q = 1
        def transform(x):
            return (x - min_) / q
        def inv_transform(x):
            return x * q + min_
    return Scaler(transform=transform, inv_transform=inv_transform) 




#根据模型最大上下文长度截断输入数据
def truncate_input(input_arr, input_str, settings, model, steps):
    """
    根据模型的最大上下文长度截断输入数据。

    参数:
    - input_arr (array-like): 输入时间序列数据。
    - input_str (str): 序列化的输入时间序列数据。
    - settings (SerializerSettings): 序列化设置。
    - model (str): 使用的LLM模型名称。
    - steps (int): 预测的步数。

    返回:
    - tuple: 包含截断后的输入时间序列数据和序列化字符串的元组。
    """
    # 如果模型支持分词和上下文长度限制，则进行截断
    if model in tokenization_fns and model in context_lengths:
        tokenization_fn = tokenization_fns[model]
        context_length = context_lengths[model]
        input_str_chuncks = input_str.split(settings.time_sep)
        
        # 从后向前截断输入字符串，直到满足上下文长度限制
        for i in range(len(input_str_chuncks) - 1):
            truncated_input_str = settings.time_sep.join(input_str_chuncks[i:])
            if not truncated_input_str.endswith(settings.time_sep):
                truncated_input_str += settings.time_sep

            # 计算输入和输出的token数量
            input_tokens = tokenization_fn(truncated_input_str)
            num_input_tokens = len(input_tokens)
            avg_token_length = num_input_tokens / (len(input_str_chuncks) - i)
            num_output_tokens = avg_token_length * steps * STEP_MULTIPLIER

            # 如果总token数不超过上下文长度，则停止截断
            if num_input_tokens + num_output_tokens <= context_length:
                truncated_input_arr = input_arr[i:]
                break
        if i > 0:
            print(f'警告: 输入数据从 {len(input_arr)} 截断到 {len(truncated_input_arr)}')
        return truncated_input_arr, truncated_input_str
    else:
        # 如果模型不支持截断，则返回原始数据
        return input_arr, input_str



def handle_prediction(pred, expected_length, strict=False):
    """
    处理LLM输出的预测结果，确保其长度符合预期。

    参数:
    - pred (array-like or None): LLM的预测结果，None表示反序列化失败。
    - expected_length (int): 预期的预测长度。
    - strict (bool, optional): 如果为True，则对无效预测返回None，默认为False。

    返回:
    - array-like: 处理后的预测结果。
    """
    if pred is None:
        return None
    else:
        if len(pred) < expected_length:
            if strict:
                print(f'警告: 预测结果过短 {len(pred)} < {expected_length}, 返回None')
                return None
            else:
                print(f'警告: 预测结果过短 {len(pred)} < {expected_length}, 使用最后一个值填充')
                return np.concatenate([pred, np.full(expected_length - len(pred), pred[-1])])
        else:
            return pred[:expected_length]



#对LLM进行文本补全，返回数值预测
def generate_predictions(
    completion_fn, 
    input_strs, 
    steps, 
    settings: SerializerSettings, 
    scalers=None,
    num_samples=1, 
    temp=0.7, 
    parallel=True,
    strict_handling=False,
    max_concurrent=10,
    **kwargs
):
    """

    参数:
    - completion_fn (callable): 用于从LLM获取文本补全的函数。
    - input_strs (list of array-like): 输入时间序列列表。
    - steps (int): 预测的步数。
    - settings (SerializerSettings): 序列化设置。
    - scalers (list of Scaler, optional): 缩放器列表，默认为None，表示不进行缩放。
    - num_samples (int, optional): 返回的样本数量，默认为1。
    - temp (float, optional): 采样温度，默认为0.7。
    - parallel (bool, optional): 如果为True，则并行运行补全任务，默认为True。
    - strict_handling (bool, optional): 如果为True，则对不符合格式或长度的预测返回None，默认为False。
    - max_concurrent (int, optional): 最大并发补全任务数，默认为10。
    - **kwargs: 其他关键字参数。

    返回:
    - tuple: 包含数值预测、原始文本补全结果和输入字符串的元组。
    """
    completions_list = []
    complete = lambda x: completion_fn(input_str=x, steps=steps * STEP_MULTIPLIER, settings=settings, num_samples=num_samples, temp=temp)
    
    # 并行或串行运行补全任务
    if parallel and len(input_strs) > 1:
        print('并行运行补全任务')
        with ThreadPoolExecutor(min(max_concurrent, len(input_strs))) as p:
            completions_list = list(tqdm(p.map(complete, input_strs), total=len(input_strs)))
    else:
        completions_list = [complete(input_str) for input_str in tqdm(input_strs)]
    
    # 将补全结果转换为数值预测
    def completion_to_pred(completion, inv_transform): 
        pred = handle_prediction(deserialize_str(completion, settings, ignore_last=False, steps=steps), expected_length=steps, strict=strict_handling)
        if pred is not None:
            return inv_transform(pred)
        else:
            return None
    
    preds = [[completion_to_pred(completion, scaler.inv_transform) for completion in completions] for completions, scaler in zip(completions_list, scalers)]
    return preds, completions_list, input_strs



#    获取LLMTime模型的预测数据。
def get_llmtime_predictions_data(train, test, model, settings, num_samples=10, temp=0.7, alpha=0.95, beta=0.3, basic=False, parallel=True, **kwargs):
    """
    参数:
    - train: 训练数据。
    - test: 测试数据。
    - model: 使用的LLM模型名称。
    - settings: 序列化设置。
    - num_samples: 生成的样本数量。
    - temp: 采样温度。
    - alpha: 缩放分位数。
    - beta: 偏移参数。
    - basic: 是否使用基础缩放。
    - parallel: 是否并行运行补全任务。
    - **kwargs: 其他关键字参数。

    返回:
    - out_dict: 包含预测样本、中位数、NLL/D等信息的字典。
    """
    # 检查模型和设置
    print("传入的model:", model)
    print("传入的setting:", settings)

    # 检查模型是否支持
    assert model in completion_fns, f'无效的模型 {model}, 必须是 {list(completion_fns.keys())} 之一'
    completion_fn = completion_fns[model]
    assert model in context_lengths, f'模型 {model} 的上下文长度未定义'
    nll_fn = nll_fns[model] if model in nll_fns else None

    # 如果设置是字典，则转换为SerializerSettings对象
    if isinstance(settings, dict):
        settings = SerializerSettings(**settings)
    if not isinstance(train, list):
        # 假设是单组训练/测试数据
        train = [train]
        test = [test]

    # 将数据转换为pandas.Series
    for i in range(len(train)):
        if not isinstance(train[i], pd.Series):
            train[i] = pd.Series(train[i], index=pd.RangeIndex(len(train[i])))
            test[i] = pd.Series(test[i], index=pd.RangeIndex(len(train[i]), len(test[i]) + len(train[i])))

    # 确保所有测试数据的长度相同
    test_len = len(test[0])
    assert all(len(t) == test_len for t in test), f'所有测试序列的长度必须相同，当前长度: {[len(t) for t in test]}'

    # 生成缩放器
    scalers = [get_scaler(train[i].values, alpha=alpha, beta=beta, basic=basic) for i in range(len(train))]

    # 对输入数据进行缩放和序列化
    input_arrs = [train[i].values for i in range(len(train))]
    transformed_input_arrs = np.array([scaler.transform(input_array) for input_array, scaler in zip(input_arrs, scalers)])
    input_strs = [serialize_arr(scaled_input_arr, settings) for scaled_input_arr in transformed_input_arrs]

    # 根据模型的最大上下文长度截断输入数据
    input_arrs, input_strs = zip(*[truncate_input(input_array, input_str, settings, model, test_len) for input_array, input_str in zip(input_arrs, input_strs)])
    
    steps = test_len
    samples = None
    medians = None
    completions_list = None

    # 生成预测样本
    if num_samples > 0:
        preds, completions_list, input_strs = generate_predictions(completion_fn, input_strs, steps, settings, scalers,
                                                                    num_samples=num_samples, temp=temp, 
                                                                    parallel=parallel, **kwargs)
        samples = [pd.DataFrame(preds[i], columns=test[i].index) for i in range(len(preds))]
        medians = [sample.median(axis=0) for sample in samples]
        samples = samples if len(samples) > 1 else samples[0]
        medians = medians if len(medians) > 1 else medians[0]

    # 构建输出字典
    out_dict = {
        'samples': samples,
        'median':  medians,
        'info': {
            'Method': model,
        },
        'completions_list': completions_list,
        'input_strs': input_strs,
    }

    # 计算NLL/D
    if nll_fn is not None:
        BPDs = [nll_fn(input_arr=input_arrs[i], target_arr=test[i].values, settings=settings, transform=scalers[i].transform, count_seps=True, temp=temp) for i in range(len(train))]
        out_dict['NLL/D'] = np.mean(BPDs)
    return out_dict


