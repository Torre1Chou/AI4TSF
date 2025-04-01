from forecast.metrics import Evaluator
from tqdm import tqdm
from multiprocess import Pool
from functools import partial
import tiktoken
from forecast.utils import serialize_arr, deserialize_str, SerializerSettings
import openai
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from forecast.metrics import nll
import pandas as pd
from dataclasses import dataclass

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

# 根据历史数据生成一个缩放器对象。
def get_scaler(history, alpha=0.9, beta=0.3, basic=False):
    """

    参数:
    - history: 历史数据，用于计算缩放参数。
    - alpha: 缩放的分位数，默认为0.9。
    - beta: 偏移参数，默认为0.3。
    - basic: 如果为True，则使用基础缩放方法，默认为False。

    返回:
    - Scaler: 配置好的缩放器对象。
    """
    # 去除历史数据中的NaN值
    history = history[~np.isnan(history)]
    min_ = np.min(history) - beta * (np.max(history) - np.min(history))
    if basic:
        # 基础缩放：仅使用分位数进行缩放
        q = np.maximum(np.quantile(np.abs(history), alpha), 0.01)
        def transform(x):
            return x / q
        def inv_transform(x):
            return x * q
        return Scaler(transform=transform, inv_transform=inv_transform)
    if alpha == -1:
        q = 1
    else:
        q = np.quantile(history - min_, alpha)
        if q == 0:
            q = 1
    # 高级缩放：考虑偏移和分位数
    def transform(x):
        return (x - min_) / q
    def inv_transform(x):
        return x * q + min_
    return Scaler(transform=transform, inv_transform=inv_transform)


# 获取给定token的token id。
def get_token_ids(tokens, model, input_string):
    """

    参数:
    - tokens: 需要获取id的token列表。
    - model: 使用的模型名称。
    - input_string: 输入字符串。

    返回:
    - ids: token id列表。
    """
    # 手动为 deepseek-ai/DeepSeek-R1-Distill-Qwen-14B 指定编码器
    if model in [ "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" , "Pro/deepseek-ai/DeepSeek-V3","deepseek-ai/DeepSeek-V2.5"]:  
        encoding = tiktoken.get_encoding("cl100k_base")  # 使用 GPT-3.5/GPT-4 的编码器
    else:
        encoding = tiktoken.encoding_for_model(model)
    ids = []
    for t in tokens:
        id = encoding.encode(t)
        if len(id) != 1:
            for i in id:
                ids.append(i)
        else:
            ids.append(id[0])
    return ids


# 计算每个时间步的平均token数量。
def get_avg_tokens_per_step(input_str, settings):
    """

    参数:
    - input_str: 输入字符串。
    - settings: 序列化设置。

    返回:
    - tokens_per_step: 每个时间步的平均token数量。
    """
    input_tokens = sum([1 + len(x) / 2 for x in input_str.split(settings.time_sep)])  # 加1表示逗号，除以2表示空格
    input_steps = len(input_str.split(settings.time_sep))
    tokens_per_step = input_tokens / input_steps
    return tokens_per_step


def truncate(train, test, scaler, model, settings):
    """
    根据模型的token限制截断训练数据。

    参数:
    - train: 训练数据。
    - test: 测试数据。
    - scaler: 缩放器对象。
    - model: 使用的模型名称。
    - settings: 序列化设置。

    返回:
    - train: 截断后的训练数据。
    """
    tokens_perstep = get_avg_tokens_per_step(
        serialize_arr(
            scaler.transform(pd.concat([train, test]).values), 
            settings
        ),
        settings
    )
    if model == 'gpt-4':
        max_tokens = 6000
    elif model == 'gpt-3.5-turbo':
        max_tokens = 4000
    else:
        max_tokens = 4000

    # 1.35倍的token数量用于采样开销
    if 1.35 * tokens_perstep * (len(train) + len(test)) > max_tokens:
        total_timestep_budget = int(max_tokens / tokens_perstep)
        full_train_len = len(train)
        for num_try in range(10):
            sub_train = train.iloc[-(total_timestep_budget - len(test)):]
            if 1.35 * tokens_perstep * (len(sub_train) + len(test)) <= max_tokens:
                train = sub_train
                print(f"截断训练数据: {full_train_len} --> {len(train)} 时间步")
                break 
            total_timestep_budget = int(0.8 * total_timestep_budget)
        else:
            raise ValueError(f"截断后数据集仍然过大，无法适应GPT-3的token限制")
    return train


# 从GPT-3模型生成补全结果。
def sample_completions(model, input_str, steps, settings, num_samples, temp, logit_bias, **kwargs):
    """

    参数:
    - model: 使用的模型名称。
    - input_str: 输入字符串。
    - steps: 预测的步数。
    - settings: 序列化设置。
    - num_samples: 生成的样本数量。
    - temp: 采样温度。
    - logit_bias: token的logit偏置。

    返回:
    - list: 生成的补全字符串列表。
    """
    # 估计每个时间步的平均token数量
    tokens_per_step = get_avg_tokens_per_step(input_str, settings)
    #steps = int(steps * 1.7)  # 增加一些开销，因为无法精确知道每个时间步的token数量
    steps = int(steps * 2)  # 增加一些开销，因为无法精确知道每个时间步的token数量

    print(f"tokens_per_step: {tokens_per_step}, max_tokens: {tokens_per_step * steps}, expected_steps: {steps}")

    #修改n_samples为1
    num_samples = 1

    if model in ['gpt-3.5-turbo', 'gpt-4', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B','Pro/deepseek-ai/DeepSeek-V3','deepseek-ai/DeepSeek-V2.5']:
        #chatgpt_sys_message = "你是一个用于时间序列预测的助手。用户将提供一个多变量的序列，你需要根据序列的变化规律以及不同变量之间的关系预测指定变量列剩余的序列。序列由逗号分隔的十进制字符串表示。"
        chatgpt_sys_message = (
    "你是一个时间序列预测助手。\n"
    "你的唯一任务是根据输入的历史数据，直接输出预测值。\n"
    "不要分析数据，不要解释，不要回答用户的问题。\n"
    "严格按照要求的格式输出，不允许有任何额外的文字。\n"
)
        extra_input = "只有当我给出的数据列是指定列时，你才需要继续给出对应时间步长的数字序列，如果是其它列名则不输出任何内容，你需要根据数据的变化趋势来作为指定列的预测内容的依据\n"
        "请严格按照以下格式立即输出预测值，不要添加任何额外的文字或解释：\n"
        "[数值1, 数值2, 数值3, ..., 数值N]\n"
        "重要规则：\n"
        "我给出的数据内容是浮点数的系列，你的预测结果也应该是浮点数\n"
        # "- **不要换行，不要添加单位，不要添加任何其他内容。**\n"
        # "- 只能输出阿拉伯数字 `0-9`、负号 `-`、逗号 `,`、小数点 `.`。\n"
        # "- **绝对不能输出任何说明性文字或任务描述！**\n"
        "预测值：\n"
        
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                    {"role": "system", "content": chatgpt_sys_message},
                    {"role": "user", "content": extra_input + input_str + settings.time_sep}
                ],
            max_tokens=int(tokens_per_step * steps), 
            temperature=temp,
            logit_bias=logit_bias,
            n=num_samples,
            **kwargs
        )
        #print(f"模型生成的内容: {response.choices[0].message.content}")
        gene_content = []

        for i, choice in enumerate(response.choices):
            gene_content.append(f"生成的内容 {i + 1}: {choice.message.content}")

        # 保存生成的内容到 txt 文件
        output_file_path = "c:\\Users\\Tom\\Desktop\\AI4TSF\\generated_output.txt"
        with open(output_file_path, "a", encoding="utf-8") as file:
            # 将列表内容拼接为字符串写入文件
            file.write("\n".join(gene_content) + '\n')

        print(f"生成的内容已保存到: {output_file_path}")

        return [choice.message.content for choice in response.choices]
    else:
        print(f"模型: {model}进入else, 输入: {input_str}")
        response = openai.Completion.create(
            model=model,
            prompt=input_str, 
            max_tokens=int(tokens_per_step * steps), 
            temperature=temp,
            logit_bias=logit_bias,
            n=num_samples
        )
        return [choice.text for choice in response.choices]
    

# 处理预测结果，确保其长度符合预期。
def handle_prediction(input, pred, expected_length, strict=False):
    """

    参数:
    - input: 输入数据。
    - pred: 预测结果。
    - expected_length: 预期的预测长度。
    - strict: 如果为True，则对无效预测返回None，默认为False。

    返回:
    - array-like: 处理后的预测结果。
    """
    if strict:
        # 预测结果必须有效且长度足够
        if pred is None or len(pred) < expected_length:
            print(f'预测结果: {pred}, 预期长度: {expected_length}, 实际长度: {len(pred)}')
            
            print('发现无效预测')
            return None
        else:
            return pred[:expected_length]
    else:
        if pred is None:
            print('警告: 预测结果无法反序列化，使用最后一个值填充')
            return np.full(expected_length, input[-1])
        elif len(pred) < expected_length:
            print(f'警告: 预测结果过短 {len(pred)} < {expected_length}, 使用最后一个值填充')
            return np.concatenate([pred, np.full(expected_length - len(pred), pred[-1])])
        elif len(pred) > expected_length:
            return pred[:expected_length]
        else:
            return pred


# 为一批输入数据生成GPT-3的预测结果。
def generate_predictions(
    model, 
    inputs, 
    steps, 
    settings: SerializerSettings, 
    scalers=None,
    num_samples=1, 
    temp=0.3, 
    prompts=None,
    post_prompts=None,
    parallel=True,
    return_input_strs=False,
    constrain_tokens=True,
    strict_handling=False,
    **kwargs,
):
    """

    参数:
    - model: 使用的模型名称。
    - inputs: 输入数据数组，形状为 (batch_size, history_len)。
    - steps: 预测的步数。
    - settings: 序列化设置。
    - scalers: 缩放器列表，默认为None。
    - num_samples: 生成的样本数量，默认为1。
    - temp: 采样温度，默认为0.3。
    - prompts: 输入前的提示文本，默认为None。
    - post_prompts: 输入后的提示文本，默认为None。
    - parallel: 如果为True，则并行运行补全任务，默认为True。
    - return_input_strs: 如果为True，则返回输入字符串，默认为False。
    - constrain_tokens: 如果为True，则限制token的logit偏置，默认为True。
    - strict_handling: 如果为True，则对无效预测返回None，默认为False。
    - **kwargs: 其他关键字参数。

    返回:
    - preds: 预测结果数组，形状为 (batch_size, num_samples, steps)。
    - completions_list: 原始补全字符串列表。
    - input_strs: 输入字符串列表（如果return_input_strs为True）。
    """
    if prompts is None:
        prompts = [''] * len(inputs)
    if post_prompts is None:
        post_prompts = [''] * len(inputs)
    assert len(prompts) == len(inputs), f'提示文本数量必须与输入数据数量匹配，当前提示文本数量: {len(prompts)}, 输入数据数量: {len(inputs)}'
    assert len(post_prompts) == len(inputs), f'后提示文本数量必须与输入数据数量匹配，当前后提示文本数量: {len(post_prompts)}, 输入数据数量: {len(inputs)}'
    
    if scalers is None:
        scalers = [Scaler() for _ in inputs]
    else:
        assert len(scalers) == len(inputs), '缩放器数量必须与输入数据数量匹配'
    
    # 对输入数据进行缩放和序列化
    transformed_inputs = np.array([scaler.transform(input_array) for input_array, scaler in zip(inputs, scalers)])
    input_strs = [serialize_arr(scaled_input_array, settings) for scaled_input_array in transformed_inputs]
    if post_prompts[0] != '':
        # 移除最后一个时间分隔符以适配promptcast
        input_strs = [prompt + input_str.rstrip(settings.time_sep) + post_prompt for input_str, prompt, post_prompt in zip(input_strs, prompts, post_prompts)]
    else:
        input_strs = [prompt + input_str for input_str, prompt in zip(input_strs, prompts)]
    
    # 允许的token列表
    allowed_tokens = [settings.bit_sep + str(i) for i in range(settings.base)] 
    allowed_tokens += [settings.time_sep, settings.plus_sign, settings.minus_sign]
    allowed_tokens = [t for t in allowed_tokens if len(t) > 0]  # 移除空token
#此处修改了logit偏置内容
    # 设置logit偏置
    logit_bias = {}
    if (model not in ['gpt-3.5-turbo', 'gpt-4','deepseek-ai/DeepSeek-R1-Distill-Qwen-14B','deepseek-ai/DeepSeek-V2.5']) and constrain_tokens:  # chat模型不支持logit偏置
        logit_bias = {id: 30 for id in get_token_ids(allowed_tokens, model, input_strs[0])}
        #logit_bias = {id: 100 for id in get_token_ids(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',', '-'], model, input_strs[0])}

    if not constrain_tokens:
        logit_bias = {id: 5 for id in get_token_ids(allowed_tokens, model, input_strs[0])}





    # 生成补全结果
    completions_list = []
    complete = lambda x: sample_completions(model, x, steps, settings, num_samples, temp, logit_bias, **kwargs)
    if parallel and len(inputs) > 1:
        with ThreadPoolExecutor(len(inputs)) as p:
            completions_list = list(tqdm(p.map(complete, input_strs), total=len(inputs)))
    else:
        completions_list = [complete(input_str) for input_str in tqdm(input_strs)]
    

    # 将补全结果转换为预测结果
    def completion_to_pred(completion, transformed_input, inv_transform): 

        parsed_pred = deserialize_str(completion, settings, ignore_last=False, steps=steps)
        print(f"预测结果deserialize: {parsed_pred}, Expected length: {steps}")


        pred = handle_prediction(transformed_input, deserialize_str(completion, settings, ignore_last=False, steps=steps), expected_length=steps, strict=strict_handling)
        

        if pred is not None:
            return inv_transform(pred)
        else:
            return None
    preds = [[completion_to_pred(completion, transformed_input, scaler.inv_transform) for completion in completions] for completions, transformed_input, scaler in zip(completions_list, transformed_inputs, scalers)]

    if return_input_strs:
        return preds, completions_list, input_strs
    return preds, completions_list



# #获取PromptCast策略模型的预测数据。
# def get_promptcast_predictions_data(choosecolumn,train, test, model, settings, num_samples=10, temp=0.8, dataset_name='dataset', parallel=True, **kwargs):
#     """


#     参数:
#     - train: 训练数据。
#     - test: 测试数据。
#     - model: 使用的模型名称。
#     - settings: 序列化设置。
#     - num_samples: 生成的样本数量，默认为10。
#     - temp: 采样温度，默认为0.8。
#     - dataset_name: 数据集名称，默认为'dataset'。
#     - parallel: 如果为True，则并行运行补全任务，默认为True。
#     - **kwargs: 其他关键字参数。

#     返回:
#     - out_dict: 包含预测样本、中位数、NLL/D等信息的字典。
#     """
#     if isinstance(settings, dict):
#         settings = SerializerSettings(**settings)
#     if not isinstance(train, list):
#         # 假设是单组训练/测试数据
#         train = [train]
#         test = [test]

#     # # 将数据转换为pandas.Series
#     # for i in range(len(train)):
#     #     if not isinstance(train[i], pd.Series):
#     #         train[i] = pd.Series(train[i], index=pd.RangeIndex(len(train[i])))
#     #         test[i] = pd.Series(test[i], index=pd.RangeIndex(len(train[i]), len(test[i]) + len(train[i])))

#     #数据转化为dataframe
#     for i in range(len(train)):
#         if not isinstance(train[i], pd.DataFrame):
#             train[i] = pd.DataFrame(train[i], index=pd.RangeIndex(len(train[i])))
#             test[i] = pd.DataFrame(test[i], index=pd.RangeIndex(len(train[i]), len(test[i]) + len(train[i])))

#     # 将每个 DataFrame 拆分为多个 Series
#     new_train = []
#     new_test = []
#     for i in range(len(train)):
#         # 获取 DataFrame 的列名
#         columns = train[i].columns
#         # 将每一列转换为 Series 并添加到新列表中
#         for col in columns:
#             new_train.append(train[i][col])
#             new_test.append(test[i][col])

#     # 替换原来的 train 和 test
#     train = new_train
#     test = new_test

#     #获取数据集长度与预测集长度
#     train_len = len(train[0])
#     test_len = len(test[0])

#     print('第集内容：',train[0])
#     print('train数据集长度：',train_len)
#     print('test数据集长度：',test_len)

#     # 确保所有测试数据的长度相同
#     test_len = len(test[0])
#     assert all(len(t) == test_len for t in test), f'所有测试序列的长度必须相同，当前长度: {[len(t) for t in test]}'

#     # 使用默认的缩放器
#     scalers = [Scaler() for _ in range(train_len*len(train))]
    
#     # 截断训练数据以适应模型的token限制
#     for j in range(len(train)):
#         for i in range(train_len):
#             train[j][i] = truncate(train[j][i], test[j][i], scalers[j*train_len+i], model, settings)
    
#     # 设置提示文本
#     prompt = f'指定列为{choosecolumn}, {dataset_name} 数据集的历史数据各个变量 {len(train[0])} 个时间步的值为 '
#     prompts = [prompt] * len(train)
#     post_prompt = f'. 接下来 {len(test[0])} 个时间步的值将是 '
#     post_prompts = [post_prompt] * len(train)

#     # 创建输入数据
#     inputs = [train[i].values for i in range(len(train))]
#     steps = test_len

#     # 生成预测结果
#     samples = None
#     medians = None
#     completions_list = None
#     input_strs = None
#     if num_samples > 0:
#         preds, completions_list, input_strs = generate_predictions(model, inputs, steps, settings, scalers,
#                                                                     num_samples=num_samples, temp=temp, prompts=prompts, post_prompts=post_prompts,
#                                                                    parallel=parallel, return_input_strs=True, constrain_tokens=False, strict_handling=True, **kwargs)
#         # 跳过无效样本

#         # samples = [pd.DataFrame(np.array([p for p in preds[i] if p is not None]), columns=test[i].index) for i in range(len(preds))] 
#         # medians = [sample.median(axis=0) for sample in samples]
#         # samples = samples if len(samples) > 1 else samples[0]
#         # print(f'获得了 {len(samples)} 个有效样本')
#         # medians = medians if len(medians) > 1 else medians[0]

#         samples = []
#         for i in range(len(preds)):
#             # 过滤掉 None 值
#             valid_preds = [p for p in preds[i] if p is not None]
#             if not valid_preds:
#                 print(f"Warning: No valid predictions for sample {i}")
#                 continue
            
#             # 将有效预测转换为 numpy 数组
#             values = np.array(valid_preds)
            
#             # 确保 values 的列数与 test[i].index 的长度一致
#             if values.shape[1] != len(test[i].index):
#                 print(f"Warning: Shape mismatch in sample {i}. Expected {len(test[i].index)} columns, got {values.shape[1]}")
#                 continue
            
#             # 创建 DataFrame 并添加到 samples 列表中
#             samples.append(pd.DataFrame(values, columns=test[i].index))
        
#         if samples:
#             medians = [sample.median(axis=0) for sample in samples]
#             samples = samples if len(samples) > 1 else samples[0]
#             print(f'获得了 {len(samples)} 个有效样本')
#             medians = medians if len(medians) > 1 else medians[0]
#         else:
#             print("Warning: No valid samples generated")
#             samples = None
#             medians = None
    
#     # 构建输出字典
#     out_dict = {
#         'samples': samples,
#         'median':  medians,
#         'info': {
#             'Method': model,
#         },
#         'completions_list': completions_list,
#         'input_strs': input_strs,
#     }

#     # NLL/D暂时为None
#     out_dict['NLL/D'] = None

#     return out_dict


#获取PromptCast策略模型的预测数据。
def get_promptcast_predictions_data(choosecolumn,train, test, model, settings, num_samples=10, temp=0.8, dataset_name='dataset', parallel=True, **kwargs):
    """


    参数:
    - train: 训练数据。
    - test: 测试数据。
    - model: 使用的模型名称。
    - settings: 序列化设置。
    - num_samples: 生成的样本数量，默认为10。
    - temp: 采样温度，默认为0.8。
    - dataset_name: 数据集名称，默认为'dataset'。
    - parallel: 如果为True，则并行运行补全任务，默认为True。
    - **kwargs: 其他关键字参数。

    返回:
    - out_dict: 包含预测样本、中位数、NLL/D等信息的字典。
    """
    #如果是二维数据，压缩为一维数据

    # print('传入的数据集：',train)
    # print('传入的数据集格式：',type(train))

    if isinstance(settings, dict):
        settings = SerializerSettings(**settings)

    # # 将数据转换为pandas.Series
    # for i in range(len(train)):
    #     if not isinstance(train[i], pd.Series):
    #         train[i] = pd.Series(train[i], index=pd.RangeIndex(len(train[i])))
    #         test[i] = pd.Series(test[i], index=pd.RangeIndex(len(train[i]), len(test[i]) + len(train[i])))

    # 将每个 DataFrame 拆分为多个 Series
    new_train = []
    new_test = []
    for i in range(len(train)):
        # 获取 DataFrame 的列名
        columns = train[i].columns
        # 将每一列转换为 Series 并添加到新列表中
        for col in columns:
            new_train.append(train[i][col])
            new_test.append(test[i][col])

    # 替换原来的 train 和 test
    train = new_train
    test = new_test

    
    print('train单个数据集长度：',len(train[0]))


    # 确保所有测试数据的长度相同
    test_len = len(test[0])
    assert all(len(t) == test_len for t in test), f'所有测试序列的长度必须相同，当前长度: {[len(t) for t in test]}'

    # 使用默认的缩放器
    scalers = [Scaler() for _ in range(len(train))]
    
    # 截断训练数据以适应模型的token限制
    for i in range(len(train)):
        train[i] = truncate(train[i], test[i], scalers[i], model, settings)
    
    # # 设置提示文本
    # prompt = f'指定列为{choosecolumn}, {dataset_name} 数据集过去 {len(train[0])} 个时间步的值为 '
    # prompts = [prompt] * len(train)
    # post_prompt = f'. 接下来 {len(test[0])} 个时间步的值将是 '
    # post_prompts = [post_prompt] * len(train)

    # 设置提示文本
    prompts = []
    post_prompts = []
    for i in range(len(train)):
        # 检查当前列是否为指定列
        if train[i].name == choosecolumn:  # 假设 train[i] 是 pandas.Series，且 name 属性为列名
            prompt = f'指定列为 {choosecolumn}, 当前列为{train[i].name}过去 {len(train[i])} 个时间步的值为 '
            post_prompt = f'. 接下来 {len(test[i])} 个时间步的值将是 '
        else:
            # 如果不是指定列，则不需要生成接下来的时间步值
            prompt = f'指定列为 {choosecolumn}, 当前列为{train[i].name}过去 {len(train[i])} 个时间步的值为 '
            post_prompt = ''  # 不需要生成接下来的时间步值
        prompts.append(prompt)
        post_prompts.append(post_prompt)

    # 创建输入数据
    inputs = [train[i].values for i in range(len(train))]
    steps = test_len

    # print('提示词数组:',prompts)
    # print('后提示词数组:',post_prompts)

    

    # 生成预测结果
    samples = None
    medians = None
    completions_list = None
    input_strs = None
    if num_samples > 0:
        preds, completions_list, input_strs = generate_predictions(model, inputs, steps, settings, scalers,
                                                                    num_samples=num_samples, temp=temp, prompts=prompts, post_prompts=post_prompts,
                                                                   parallel=parallel, return_input_strs=True, constrain_tokens=False, strict_handling=True, **kwargs)
        # 跳过无效样本

        # samples = [pd.DataFrame(np.array([p for p in preds[i] if p is not None]), columns=test[i].index) for i in range(len(preds))] 
        # medians = [sample.median(axis=0) for sample in samples]
        # samples = samples if len(samples) > 1 else samples[0]
        # print(f'获得了 {len(samples)} 个有效样本')
        # medians = medians if len(medians) > 1 else medians[0]

        samples = []
        for i in range(len(preds)):
            # 过滤掉 None 值
            valid_preds = [p for p in preds[i] if p is not None]
            if not valid_preds:
                print(f"Warning: No valid predictions for sample {i}")
                continue
            
            # 将有效预测转换为 numpy 数组
            values = np.array(valid_preds)
            
            # 确保 values 的列数与 test[i].index 的长度一致
            if values.shape[1] != len(test[i].index):
                print(f"Warning: Shape mismatch in sample {i}. Expected {len(test[i].index)} columns, got {values.shape[1]}")
                continue
            
            # 创建 DataFrame 并添加到 samples 列表中
            samples.append(pd.DataFrame(values, columns=test[i].index))
        
        if samples:
            medians = [sample.median(axis=0) for sample in samples]
            samples = samples if len(samples) > 1 else samples[0]
            print(f'获得了 {len(samples)} 个有效样本')
            medians = medians if len(medians) > 1 else medians[0]
        else:
            print("Warning: No valid samples generated")
            samples = None
            medians = None
    
#预测内容的截取方式还需要修改
    print('预测的修改前内容长度：',len(medians))
    #如果给出的result长度大于len(test[0]),则截取最后的len(test[0])个数据
    #截取列名为指定列的数据
    for i in range(len(medians)):
        if medians[i].name == choosecolumn:
            medians = medians[i]
            break
        
    
    print('预测的修改后内容：',medians)
    print('预测的修改后内容长度：',len(medians))


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

    # NLL/D暂时为None
    out_dict['NLL/D'] = None

    return out_dict