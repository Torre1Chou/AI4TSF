from functools import partial
from models.gpt import gpt_completion_fn, gpt_nll_fn
from models.gpt import tokenize_fn as gpt_tokenize_fn
from models.llama import llama_completion_fn, llama_nll_fn
from models.llama import tokenize_fn as llama_tokenize_fn


# 每个模型的文本补全函数
# -----------------------------------------------
# 每个模型映射到一个函数，用于生成文本补全。
# 补全函数如下：
#
# 参数:
#   - input_str (str): 输入时间序列的字符串表示。
#   - steps (int): 预测的步数。
#   - settings (SerializerSettings): 序列化设置。
#   - num_samples (int): 采样的补全数量。
#   - temp (float): 控制模型输出随机性的温度参数。
#
# 返回:
#   - list: 从模型采样的补全字符串列表。
completion_fns = {
    'text-davinci-003': partial(gpt_completion_fn, model='text-davinci-003'),
    'gpt-4': partial(gpt_completion_fn, model='gpt-4'),
    'gpt-4-1106-preview':partial(gpt_completion_fn, model='gpt-4-1106-preview'),
    'gpt-3.5-turbo-instruct': partial(gpt_completion_fn, model='gpt-3.5-turbo-instruct'),
    'llama-7b': partial(llama_completion_fn, model='7b'),
    'llama-13b': partial(llama_completion_fn, model='13b'),
    'llama-70b': partial(llama_completion_fn, model='70b'),
    'llama-7b-chat': partial(llama_completion_fn, model='7b-chat'),
    'llama-13b-chat': partial(llama_completion_fn, model='13b-chat'),
    'llama-70b-chat': partial(llama_completion_fn, model='70b-chat'),
    'deepseek-ai/DeepSeek-R1-Distill-Llama-8B': partial(gpt_completion_fn,model='deepseek-ai/DeepSeek-R1-Distill-Llama-8B'),
    
}

# 每个模型的负对数似然（NLL/D）函数（可选）
# -----------------------------------------------
# 每个模型映射到一个函数，用于计算连续负对数似然（NLL/D）。
# 该函数仅用于计算似然，不用于采样。
#
# NLL函数如下：
#
# 参数:
#   - input_arr (np.ndarray): 数据变换后的输入时间序列（历史数据）。
#   - target_arr (np.ndarray): 数据变换后的真实序列（未来数据）。
#   - settings (SerializerSettings): 序列化设置。
#   - transform (callable): 数据变换函数（例如缩放），用于确定雅可比因子。
#   - count_seps (bool): 如果为True，则在NLL计算中计入时间步分隔符，允许可变数字位数时需要。
#   - temp (float): 采样温度参数。
#
# 返回:
#   - float: 计算得到的 p(target_arr | input_arr) 的NLL/D。
nll_fns = {
    'text-davinci-003': partial(gpt_nll_fn, model='text-davinci-003'),
    'llama-7b': partial(llama_completion_fn, model='7b'),
    'llama-7b': partial(llama_nll_fn, model='7b'),
    'llama-13b': partial(llama_nll_fn, model='13b'),
    'llama-70b': partial(llama_nll_fn, model='70b'),
    'llama-7b-chat': partial(llama_nll_fn, model='7b-chat'),
    'llama-13b-chat': partial(llama_nll_fn, model='13b-chat'),
    'llama-70b-chat': partial(llama_nll_fn, model='70b-chat'),
    'deepseek-ai/DeepSeek-R1-Distill-Llama-8B': partial(gpt_nll_fn,model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
   

}

# 每个模型的分词函数（可选），仅在需要自动截断输入时使用。
# 分词函数如下：
#
# 参数:
#   - str (str): 需要分词的字符串。
# 返回:
#   - token_ids (list): 分词后的token id列表。
tokenization_fns = {
    'text-davinci-003': partial(gpt_tokenize_fn, model='text-davinci-003'),
    'gpt-3.5-turbo-instruct': partial(gpt_tokenize_fn, model='gpt-3.5-turbo-instruct'),
    #new
    'gpt-4': partial(gpt_tokenize_fn, model='gpt-4'),
    'llama-7b': partial(llama_tokenize_fn, model='7b'),
    'llama-13b': partial(llama_tokenize_fn, model='13b'),
    'llama-70b': partial(llama_tokenize_fn, model='70b'),
    'llama-7b-chat': partial(llama_tokenize_fn, model='7b-chat'),
    'llama-13b-chat': partial(llama_tokenize_fn, model='13b-chat'),
    'llama-70b-chat': partial(llama_tokenize_fn, model='70b-chat'),
    'deepseek-ai/DeepSeek-R1-Distill-Llama-8B':partial(gpt_tokenize_fn,model ="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
}


# 每个模型的上下文长度（可选），仅在需要自动截断输入时使用。
context_lengths = {
    'text-davinci-003': 4097,
    'gpt-3.5-turbo-instruct': 4097,
    'gpt-4': 4097,
    'llama-7b': 4096,
    'llama-13b': 4096,
    'llama-70b': 4096,
    'llama-7b-chat': 4096,
    'llama-13b-chat': 4096,
    'llama-70b-chat': 4096,
    'deepseek-ai/DeepSeek-R1-Distill-Llama-8B':4097,
}