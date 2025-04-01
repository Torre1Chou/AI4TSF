import openai



#接入聊天机器人
def chatbot(model,ask,field):
    #设置API_KEY
    deepseek_api='sk-wdkwczxanmwevdpenbguhirxnjxpoyvoaqxrekhftvuizsld'
    gpt_api='sk-J4QMUqSVow9TsSdAIABKOIclpsFSpyPh2PJXccvbzBEepliv'
    
    #设置API_BASE
    deepseek_base='https://api.siliconflow.cn/v1'
    gpt_base='https://chatapi.littlewheat.com/v1'

    #数据的领域
    datafield = field

#这里需要设置选择不同的聊天助手
    if model == 'DeepSeek':
        current_api = deepseek_api
        api_base = deepseek_base
    elif model == 'GPT-3.5':
        current_api = gpt_api
        api_base = gpt_base
    else:
        print('Model not found')
        return None



    #当button_send触发之后获取文本框中发送的信息
    current_text = ask

    openai.api_key = current_api
    openai.api_base = api_base

    #打印当前的模型，获取的文本
    print('Chat Model:',model)
    print('Current Text:',current_text)

    #根据传入的模型获取对应的模型名称
    if model == 'DeepSeek':
        model_name = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
        #model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
    elif model == 'GPT-3.5':
        model_name = 'gpt-3.5-turbo'
    else:
        print('Model not found')
        return None

#提示词的构建还需要修改
    completion = openai.ChatCompletion.create(
        model= model_name,
        messages=[
            {"role": "system", "content": f"你是一个数据分析与处理专家。文件中的数据序列来自领域{datafield}，你需要分析文件中的数据内容，并以中文回答数据特征、发展趋势以及可能的洞察。"},
            {"role": "user", "content": current_text},
        ]
    )

    #获取聊天机器人的回复
    response = completion.choices[0].message['content']
    
    #返回聊天机器人的回复
    return response

# 读取文件内容
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

#需要把载入的文件远程发送给聊天机器人    
def chatbot_file(model,file_path,field):
    #设置API_KEY
    deepseek_api='sk-mbckewojnfssrriqyaipcqorevtzaqrnrbfxhfoxoamdyxix'
    gpt_api='sk-J4QMUqSVow9TsSdAIABKOIclpsFSpyPh2PJXccvbzBEepliv'
    
    #设置API_BASE
    deepseek_base='https://api.siliconflow.cn/v1'
    gpt_base='https://chatapi.littlewheat.com/v1'

    #这里需要设置选择不同的聊天助手
    if model == 'DeepSeek':
        current_api = deepseek_api
        api_base = deepseek_base
    elif model == 'GPT-3.5':
        current_api = gpt_api
        api_base = gpt_base
    else:
        print('Model not found')
        return None
    
    current_text = read_file(file_path)

    openai.api_key = current_api
    openai.api_base = api_base

    datafield = field

    #根据传入的模型获取对应的模型名称
    if model == 'DeepSeek':
        #model_name = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
        model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
    elif model == 'GPT-3.5':
        model_name = 'gpt-3.5-turbo'
    else:
        print('Model not found')
        return None
    
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": f"你是一个数据分析与处理专家。文件中的数据序列来自领域{datafield},你需要分析文件中的数据内容，并以中文回答数据特征、发展趋势以及可能的洞察。"},
            {"role": "user", "content": f"请分析以下数据内容，并总结数据特征、发展趋势以及可能的洞察：\n{current_text}"},
        ]
    )

    response = completion.choices[0].message['content']
    return response
