import numpy as np
import numbers
import random
from collections import defaultdict
from collections.abc import Iterable

from dataclasses import is_dataclass
from typing import Any

import itertools,operator,functools

class FixedNumpySeed:
    """
    固定随机种子上下文管理器，用于确保在特定代码块中使用固定的随机种子。
    """
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        # 保存当前的随机状态
        self.np_rng_state = np.random.get_state()
        self.rand_rng_state = random.getstate()
        # 设置新的随机种子
        np.random.seed(self.seed)
        random.seed(self.seed)

    def __exit__(self, *args):
        # 恢复之前的随机状态
        np.random.set_state(self.np_rng_state)
        random.setstate(self.rand_rng_state)

class ReadOnlyDict(dict):
    """
    只读字典类，禁止对字典进行任何修改操作。
    """
    def __readonly__(self, *args, **kwargs):
        raise RuntimeError("无法修改只读字典")
    __setitem__ = __readonly__
    __delitem__ = __readonly__
    pop = __readonly__
    popitem = __readonly__
    clear = __readonly__
    update = __readonly__
    setdefault = __readonly__
    del __readonly__


class NoGetItLambdaDict(dict):
    """
    禁止获取lambda或可迭代对象的字典类。
    """
    def __init__(self, d={}):
        super().__init__()
        for k, v in d.items():
            if isinstance(v, dict):
                self[k] = NoGetItLambdaDict(v)
            else:
                self[k] = v

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if callable(value) and value.__name__ == "<lambda>":
            raise LookupError("不应从此字典中获取lambda函数")
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict, tuple)):
            raise LookupError("不应从此字典中获取可迭代对象")
        return value
    
#配置
def sample_config(config_spec):
    """

    参数:
    - config_spec: 配置规范字典。

    返回:
    - dict: 生成的配置字典。
    """
    cfg_all = config_spec
    more_work = True
    i = 0
    while more_work:
        cfg_all, more_work = _sample_config(cfg_all, NoGetItLambdaDict(cfg_all))
        i += 1
        if i > 10:
            raise RecursionError("配置依赖无法解析")
    out = defaultdict(dict)
    out.update(cfg_all)
    return out

# 递归生成配置。
def _sample_config(config_spec, cfg_all):
    """

    参数:
    - config_spec: 配置规范字典。
    - cfg_all: 当前配置字典。

    返回:
    - dict: 生成的配置字典。
    - bool: 是否需要进一步处理。
    """
    cfg = {}
    more_work = False
    for k, v in config_spec.items():
        if isinstance(v, dict):
            new_dict, extra_work = _sample_config(v, cfg_all)
            cfg[k] = new_dict
            more_work |= extra_work
        elif isinstance(v, Iterable) and not isinstance(v, (str, bytes, dict, tuple)):
            cfg[k] = random.choice(v)
        elif callable(v) and v.__name__ == "<lambda>":
            try:
                cfg[k] = v(cfg_all)
            except (KeyError, LookupError, Exception):
                cfg[k] = v  # 使用lambda函数本身而不是其返回值
                more_work = True
        else:
            cfg[k] = v
    return cfg, more_work


#将嵌套字典展平为单层字典。
def flatten(d, parent_key='', sep='/'):
    """

    参数:
    - d: 嵌套字典。
    - parent_key: 父键。
    - sep: 分隔符。

    返回:
    - dict: 展平后的字典。
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict) and v:  # 非空字典
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# 将展平的字典恢复为嵌套字典。
def unflatten(d, sep='/'):
    """

    参数:
    - d: 展平的字典。
    - sep: 分隔符。

    返回:
    - dict: 恢复后的嵌套字典。
    """
    out_dict = {}
    for k, v in d.items():
        if isinstance(k, str):
            keys = k.split(sep)
            dict_to_modify = out_dict
            for partial_key in keys[:-1]:
                try:
                    dict_to_modify = dict_to_modify[partial_key]
                except KeyError:
                    dict_to_modify[partial_key] = {}
                    dict_to_modify = dict_to_modify[partial_key]
            if keys[-1] in dict_to_modify:
                dict_to_modify[keys[-1]].update(v)
            else:
                dict_to_modify[keys[-1]] = v
        else:
            out_dict[k] = v
    return out_dict



class grid_iter:
    """
    网格迭代器类，用于遍历配置规范中的网格参数。
    """
    def __init__(self, config_spec, num_elements=-1, shuffle=True):
        self.cfg_flat = flatten(config_spec)
        is_grid_iterable = lambda v: (isinstance(v, Iterable) and not isinstance(v, (str, bytes, dict, tuple)))
        iterables = sorted({k: v for k, v in self.cfg_flat.items() if is_grid_iterable(v)}.items())
        if iterables:
            self.iter_keys, self.iter_vals = zip(*iterables)
        else:
            self.iter_keys, self.iter_vals = [], [[]]
        self.vals = list(itertools.product(*self.iter_vals))
        if shuffle:
            with FixedNumpySeed(0):
                random.shuffle(self.vals)
        self.num_elements = num_elements if num_elements >= 0 else (-1 * num_elements) * len(self)

    def __iter__(self):
        self.i = 0
        self.vals_iter = iter(self.vals)
        return self

    def __next__(self):
        self.i += 1
        if self.i > self.num_elements:
            raise StopIteration
        if not self.vals:
            v = []
        else:
            try:
                v = next(self.vals_iter)
            except StopIteration:
                self.vals_iter = iter(self.vals)
                v = next(self.vals_iter)
        chosen_iter_params = dict(zip(self.iter_keys, v))
        self.cfg_flat.update(chosen_iter_params)
        return sample_config(unflatten(self.cfg_flat))

    def __len__(self):
        product = functools.partial(functools.reduce, operator.mul)
        return product(len(v) for v in self.iter_vals) if self.vals else 1
    

# 展平字典，忽略外层键。
def flatten_dict(d):
    """
    参数:
    - d: 字典。

    返回:
    - dict: 展平后的字典。
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten_dict(v))
        elif isinstance(v, (numbers.Number, str, bytes)):
            out[k] = v
        else:
            out[k] = str(v)
    return out



#将训练集划分为训练集和验证集。
def make_validation_dataset(train, n_val, val_length):
    """

    参数:
    - train: 训练数据列表。
    - n_val: 验证样本数量。
    - val_length: 每个验证样本的长度。

    返回:
    - tuple: 训练集（去除验证部分）、验证集和验证样本数量。
    """
    assert isinstance(train, list), '训练数据应为时间序列列表'
    train_minus_val_list, val_list = [], []
    if n_val is None:
        n_val = len(train)
    for train_series in train[:n_val]:
        train_len = max(len(train_series) - val_length, 1)
        train_minus_val, val = train_series[:train_len], train_series[train_len:]
        print(f'训练集长度: {len(train_minus_val)}, 验证集长度: {len(val)}')
        train_minus_val_list.append(train_minus_val)
        val_list.append(val)
    return train_minus_val_list, val_list, n_val

#在验证集上评估超参数。

def evaluate_hyper(hyper, train_minus_val, val, get_predictions_fn):
    """
    参数:
    - hyper: 超参数字典。
    - train_minus_val: 去除验证部分的训练数据。
    - val: 验证数据。
    - get_predictions_fn: 获取预测结果的函数。

    返回:
    - float: 超参数的平均NLL/D值。
    """
    assert isinstance(train_minus_val, list) and isinstance(val, list), '训练集和验证集应为时间序列列表'
    return get_predictions_fn(train_minus_val, val, **hyper, num_samples=0)['NLL/D']


# 将对象转换为字典。
def convert_to_dict(obj: Any) -> Any:
    """

    参数:
    - obj: 任意对象。

    返回:
    - Any: 转换后的字典或对象。
    """
    if isinstance(obj, dict):
        return {k: convert_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_dict(elem) for elem in obj]
    elif is_dataclass(obj):
        return convert_to_dict(obj.__dict__)
    else:
        return obj
    

from functools import partial
import numpy as np
from dataclasses import dataclass

#将数字转换为指定基数和精度的表示。
def vec_num2repr(val, base, prec, max_val):
    """

    参数:
    - val: 数字数组。
    - base: 基数。
    - prec: 精度。
    - max_val: 最大绝对值。

    返回:
    - tuple: 符号和数字的表示。
    """
    base = float(base)
    bs = val.shape[0]
    sign = 1 * (val >= 0) - 1 * (val < 0)
    val = np.abs(val)
    max_bit_pos = int(np.ceil(np.log(max_val) / np.log(base)).item())

    before_decimals = []
    for i in range(max_bit_pos):
        digit = (val / base**(max_bit_pos - i - 1)).astype(int)
        before_decimals.append(digit)
        val = val - digit * base**(max_bit_pos - i - 1)

    before_decimals = np.stack(before_decimals, axis=-1)

    if prec > 0:
        after_decimals = []
        for i in range(prec):
            digit = (val / base**(-i - 1)).astype(int)
            after_decimals.append(digit)
            val -= digit * base**(-i - 1)

        after_decimals = np.stack(after_decimals, axis=-1)
        digits = np.concatenate([before_decimals, after_decimals], axis=-1)
    else:
        digits = before_decimals
    return sign, digits


#将指定基数和精度的表示转换回数字。
def vec_repr2num(sign, digits, base, prec, half_bin_correction=True):
    """

    参数:
    - sign: 符号数组。
    - digits: 数字表示。
    - base: 基数。
    - prec: 精度。
    - half_bin_correction: 是否应用半bin校正。

    返回:
    - np.array: 转换后的数字数组。
    """
    base = float(base)
    bs, D = digits.shape
    digits_flipped = np.flip(digits, axis=-1)
    powers = -np.arange(-prec, -prec + D)
    val = np.sum(digits_flipped / base**powers, axis=-1)

    if half_bin_correction:
        val += 0.5 / base**prec

    return sign * val

@dataclass
class SerializerSettings:
    """
    序列化设置类。

    属性:
    - base: 基数。
    - prec: 精度。
    - signed: 是否允许负数。
    - fixed_length: 是否固定长度。
    - max_val: 最大绝对值。
    - time_sep: 时间分隔符。
    - bit_sep: 位分隔符。
    - plus_sign: 正号。
    - minus_sign: 负号。
    - half_bin_correction: 是否应用半bin校正。
    - decimal_point: 小数点。
    - missing_str: 缺失值表示。
    """
    base: int = 10
    prec: int = 3
    signed: bool = True
    fixed_length: bool = False
    max_val: float = 1e7
    time_sep: str = ' ,'
    bit_sep: str = ' '
    plus_sign: str = ''
    minus_sign: str = ' -'
    half_bin_correction: bool = True
    decimal_point: str = ''
    missing_str: str = ' Nan'


# 将数组序列化为字符串。
def serialize_arr(arr, settings: SerializerSettings):
    """

    参数:
    - arr: 数组。
    - settings: 序列化设置。

    返回:
    - str: 序列化后的字符串。
    """
    assert np.all(np.abs(arr[~np.isnan(arr)]) <= settings.max_val), f"abs(arr) must be <= max_val,\
         but abs(arr)={np.abs(arr)}, max_val={settings.max_val}"
    
    if not settings.signed:
        assert np.all(arr[~np.isnan(arr)] >= 0), f"unsigned arr must be >= 0"
        plus_sign = minus_sign = ''
    else:
        plus_sign = settings.plus_sign
        minus_sign = settings.minus_sign
    
    vnum2repr = partial(vec_num2repr,base=settings.base,prec=settings.prec,max_val=settings.max_val)
    sign_arr, digits_arr = vnum2repr(np.where(np.isnan(arr),np.zeros_like(arr),arr))
    ismissing = np.isnan(arr)
    
    def tokenize(arr):
        return ''.join([settings.bit_sep+str(b) for b in arr])
    
    bit_strs = []
    for sign, digits,missing in zip(sign_arr, digits_arr, ismissing):
        if not settings.fixed_length:
            # remove leading zeros
            nonzero_indices = np.where(digits != 0)[0]
            if len(nonzero_indices) == 0:
                digits = np.array([0])
            else:
                digits = digits[nonzero_indices[0]:]
            # add a decimal point
            prec = settings.prec
            if len(settings.decimal_point):
                digits = np.concatenate([digits[:-prec], np.array([settings.decimal_point]), digits[-prec:]])
        digits = tokenize(digits)
        sign_sep = plus_sign if sign==1 else minus_sign
        if missing:
            bit_strs.append(settings.missing_str)
        else:
            bit_strs.append(sign_sep + digits)
    bit_str = settings.time_sep.join(bit_strs)
    bit_str += settings.time_sep # otherwise there is ambiguity in number of digits in the last time step
    return bit_str


#将字符串反序列化为数组。
def deserialize_str(bit_str, settings: SerializerSettings, ignore_last=False, steps=None):
    """

    参数:
    - bit_str: 字符串。
    - settings: 序列化设置。
    - ignore_last: 是否忽略最后一个时间步。
    - steps: 反序列化的步数。

    返回:
    - np.array: 反序列化后的数组。
    """

    orig_bitstring = bit_str
    bit_strs = bit_str.split(settings.time_sep)
    # remove empty strings
    bit_strs = [a for a in bit_strs if len(a) > 0]
    if ignore_last:
        bit_strs = bit_strs[:-1]
    if steps is not None:
        bit_strs = bit_strs[:steps]
    vrepr2num = partial(vec_repr2num,base=settings.base,prec=settings.prec,half_bin_correction=settings.half_bin_correction)
    max_bit_pos = int(np.ceil(np.log(settings.max_val)/np.log(settings.base)).item())
    sign_arr = []
    digits_arr = []
    try:
        for i, bit_str in enumerate(bit_strs):
            if bit_str.startswith(settings.minus_sign):
                sign = -1
            elif bit_str.startswith(settings.plus_sign):
                sign = 1
            else:
                assert settings.signed == False, f"signed bit_str must start with {settings.minus_sign} or {settings.plus_sign}"
            bit_str = bit_str[len(settings.plus_sign):] if sign==1 else bit_str[len(settings.minus_sign):]
            if settings.bit_sep=='':
                bits = [b for b in bit_str.lstrip()]
            else:
                bits = [b[:1] for b in bit_str.lstrip().split(settings.bit_sep)]
            if settings.fixed_length:
                # print(f'in deserialize: {settings.fixed_length}')
                assert len(bits) == max_bit_pos+settings.prec, f"fixed length bit_str must have {max_bit_pos+settings.prec} bits, but has {len(bits)}: '{bit_str}'"
            digits = []
            for b in bits:
                if b==settings.decimal_point:
                    continue
                # check if is a digit
                if b.isdigit():
                    digits.append(int(b))
                else:
                    break
            #digits = [int(b) for b in bits]
            sign_arr.append(sign)
            digits_arr.append(digits)
    except Exception as e:
        print(f"Error deserializing {settings.time_sep.join(bit_strs[i-2:i+5])}{settings.time_sep}\n\t{e}")
        print(f'Got {orig_bitstring}')
        print(f"Bitstr {bit_str}, separator {settings.bit_sep}")
        # At this point, we have already deserialized some of the bit_strs, so we return those below
    if digits_arr:
        # add leading zeros to get to equal lengths
        max_len = max([len(d) for d in digits_arr])
        for i in range(len(digits_arr)):
            digits_arr[i] = [0]*(max_len-len(digits_arr[i])) + digits_arr[i]
        return vrepr2num(np.array(sign_arr), np.array(digits_arr))
    else:
        # errored at first step
        return None
