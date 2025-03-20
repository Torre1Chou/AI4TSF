import os
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

from forecast.utils import make_validation_dataset
from forecast.utils import evaluate_hyper
from forecast.utils import grid_iter
from forecast.utils import convert_to_dict

from forecast.utils import SerializerSettings

from models.arima import get_arima_predictions_data
from models.llmtime import get_llmtime_predictions_data
from models.promptcast import get_promptcast_predictions_data
from models.timeGPT import get_timeGPT_predictions_data
from models.Leddam.run import get_Leddam_predictions_data

os.environ['OMP_NUM_THREADS']='4'


#设置模型超参数
arima_hypers = dict(p=[12, 30], d=[1, 2], q=[0])


# DeepSeek API 超参数设置
deepseek_hypers = dict(
    temp =1.0,  # 温度参数，控制生成文本的多样性
    top_p =0.8,  # 控制生成文本的多样性，累积概率阈值
    diversity_penalty= 0.3,  # 调整生成文本的多样性和可读性之间的平衡
    basic_mode= True,  # 使用基本生成模式
    settings= SerializerSettings ( # 序列化设置
        base= 10,  # 使用十进制
        prec= 3,  # 小数点后保留三位有效数字
        signed= True,  # 允许生成带符号数值
        time_sep= ", ",  # 时间分隔符为逗号后跟一个空格
        bit_sep= "",  # 没有比特分隔符
        minus_sign= "-",  # 使用减号作为负数的符号
    )

)

gpt3_hypers = dict(
    temp=0.7,
    alpha=0.95,  # 倾向于生成更加可读的文本
    beta=0.3,  # 控制生成文本的长度,较小的 beta 值会导致生成较短的文本
    basic=False,
    # 启用半数值二进制修正（half_bin_correction=True）
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True)
)

promptcast_hypers = dict(
    temp=0.7,
    settings=SerializerSettings(base=10, prec=0, signed=True,
                                time_sep=', ',
                                bit_sep='',
                                plus_sign='',
                                minus_sign='-',
                                half_bin_correction=False,
                                decimal_point='')
)


#后面可以修改参数
timeGPT_hypers = dict(
    temp=0.7,
    alpha=0.95,  # 倾向于生成更加可读的文本
    beta=0.3,  # 控制生成文本的长度,较小的 beta 值会导致生成较短的文本
    basic=False,
    # 启用半数值二进制修正（half_bin_correction=True）
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True)
)


#
Leddam_hypers = dict(
    
)

model_hypers = {
    'ARIMA': arima_hypers,
    'DeepSeek':{'model':'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',**promptcast_hypers},
    'LLMTime GPT-3.5': {'model': 'gpt-3.5-turbo-instruct', **gpt3_hypers},
    'PromptCast GPT-3': {'model': 'gpt-3.5-turbo-instruct', **promptcast_hypers},
    'timeGPT': {'model': 'timegpt-1-long-horizon', **timeGPT_hypers},
    'Leddam':Leddam_hypers,
}

#切换deepseek的模型
# model_hypers = {
#     'ARIMA': arima_hypers,
#     'DeepSeek':{'model':'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',**promptcast_hypers},
#     'LLMTime GPT-3.5': {'model': 'gpt-3.5-turbo-instruct', **gpt3_hypers},
#     'PromptCast GPT-3': {'model': 'gpt-3.5-turbo-instruct', **promptcast_hypers},
    
# }


#设置模型预测函数
model_predict_fns = {
    'ARIMA': get_arima_predictions_data,
    'DeepSeek':get_promptcast_predictions_data,
    'LLMTime GPT-3.5': get_llmtime_predictions_data,
    'PromptCast GPT-3': get_promptcast_predictions_data,
    'timeGPT': get_timeGPT_predictions_data,
    'Leddam':get_Leddam_predictions_data,
}




#设置模型预测函数
def common_predict_fn(origin_data, predict_data, model_fn, hyper,n_samples, verbose=False, parallel=True, n_train=None, n_val=None):
    """
    Common function for getting predictions from a model.
    """
    if isinstance(hyper, dict):
        hyper = list(grid_iter(hyper))
    else:
        assert isinstance(hyper, list), 'hyper must be a list or dict'
    if not isinstance(origin_data, list):
        origin_data = [origin_data]
        predict_data = [predict_data]

    if n_val is None:
        n_val = len(origin_data)

    if len(hyper) > 1:
        val_length = min(len(predict_data[0]), int(np.mean([len(series) for series in origin_data])/2))

        train_val,val, n_val = make_validation_dataset(origin_data, n_val=n_val, val_length=val_length) # use half of train as val for tiny train sets
        # remove validation series that has smaller length than required val_length
        train_val, val = zip(*[(origin_series, predict_series) for origin_series, predict_series in zip(train_val, val) if len(predict_series) == val_length])
        train_val = list(train_val)

        val = list(val)
        if len(train_val) <= int(0.9*n_val):
            raise ValueError(f'Removed too many validation series. Only {len(origin_data)} out of {len(n_val)} series have length >= {val_length}. Try or decreasing val_length.')
        val_nlls = []
        def eval_hyper(hyper):
            try:
                return hyper, evaluate_hyper(hyper, train_val, val, model_fn)
            except ValueError:
                return hyper, float('inf')
            
        best_val_nll = float('inf')
        best_hyper = None
        if not parallel:
            for hyper in tqdm(hyper, desc='Hyperparameter search'):
                _,val_nll = eval_hyper(hyper)
                val_nlls.append(val_nll)
                if val_nll < best_val_nll:
                    best_val_nll = val_nll
                    best_hyper = hyper
                if verbose:
                    print(f'Hyper: {hyper} \n\t Val NLL: {val_nll:3f}')
        else:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(eval_hyper,hyper) for hyper in hyper]
                for future in tqdm(as_completed(futures), total=len(hyper), desc='Hyperparameter search'):
                    hyper,val_nll = future.result()
                    val_nlls.append(val_nll)
                    if val_nll < best_val_nll:
                        best_val_nll = val_nll
                        best_hyper = hyper
                    if verbose:
                        print(f'Hyper: {hyper} \n\t Val NLL: {val_nll:3f}')
    else:
        best_hyper = hyper[0]
        best_val_nll = float('inf')

    # 如果 best_hyper 为 None，使用第一个超参数作为默认值
    if best_hyper is None:
        print("Warning: No valid hyperparameters found, using the first hyperparameter as default.")
        best_hyper = hyper[0]
    

    print(f'Sampling with best hyper... {best_hyper} \n with NLL {best_val_nll:3f}')
    
    



    out= model_fn(origin_data, predict_data, **best_hyper, num_samples=n_samples, n_train=n_train, parallel=parallel)
    
    out['best_hyper']=convert_to_dict(best_hyper)
    return out





import openai
#可选用openai的API还是deepseek的API


def get_predictions(models,API_KEY,origin_data,predict_data, hypers):
    """
    Get predictions from the specified models.
    """
    # Set up the OpenAI API
    os.environ["OPENAI_API_KEY"]= API_KEY
    openai.api_key = os.environ["OPENAI_API_KEY"]
    if(models=='DeepSeek'):
        openai.api_base= os.environ.get("OPENAI_API_BASE","https://api.siliconflow.cn/v1")
    elif(models=='LLMTime GPT-3.5'):
        openai.api_base= os.environ.get("OPENAI_API_BASE","https://chatapi.littlewheat.com/v1")
    #print("从前hypers参数：",hypers)

#后面要改成从hyers参数中获取选中的模型
    # 如果 hypers 为空，则使用默认的 model_hypers
    if hypers is None:
        hypers = model_hypers

    choose_models_hyper = model_hypers
    predict_results = {}

    #print("现在hypers参数：",hypers)
    #print("现在models参数：",choose_models_hyper)

#此处设置的模型只能有一个，所以不过循环了


#加入模型后hyper应该是一个字典，包含不同模型，在循环中选择
    #for model in models:
    if models in model_predict_fns:
        predict_fn = model_predict_fns[models]
        # 如果 hypers 中没有该模型的配置，则使用默认的 model_hypers
#后面要换回用hypers参数
        #model_hyper = list(grid_iter(hypers.get(model, model_hypers.get(model))))
        #model_hyper = list(grid_iter(choose_models_hyper.get(model, model_hypers.get(model))))
        model_hyper = list(grid_iter(choose_models_hyper[models]))

#samples参数也需要修改
        n_samples = 10


        if model_hyper is None:
            raise KeyError(f"超参数配置中缺少模型 '{models}' 的配置")
        
        # for column in origin_data.columns:
        #     # 提取单变量数据
        #     origin_single = origin_data[column]
        #     predict_single = predict_data[column]

        
#parrallel的正负对预测结果的影响没搞懂，后续要调整
        result = common_predict_fn(origin_data, predict_data, predict_fn, model_hyper,n_samples, verbose=False, parallel=True)
        #predict_results = result
    else:
        raise ValueError(f"Model '{models}' not found in model_predict_fns")

#
    #predict_results = result
    #打印result这个dict中的所有键
    #print("Result keys:", result.keys())

    # 提取预测结果的中位数
    if 'median' in result:
        predict_results = result['median']
    else:
        raise ValueError("预测结果中没有 'median' 键")
    
    #打印预测结果格式
    #print("内层函数预测结果格式：",type(result))
    #print("内层函数预测结果：",result)
    
    return predict_results

#predict_data的格式不对，要改和original_data对齐
