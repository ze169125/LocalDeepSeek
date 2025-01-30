#####                           PARTE CARREGANDO MODELO
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
#definimos onde está o modelo
model_name = "./DeepSeek-V3/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)#, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id


#####                           PARTE GENERALIZANDO
from typing import List

#inicio o historico de conversas marcando qual o prompt do sistema que irá usar sempre. Cuidado, cada modelo tem um padrão.
historico_conversas = [
    {
        "role":"system",
        "content": "You are an AI assistante that gives helpful answers on a cordial way. You must answer the questions in a short and consice way"
    }
]

def PegoResposta(entrada:str) -> str: 
    #adiciono historico até o momento
    global historico_conversas
    historico_conversas.append({"role": "user", "content":entrada})

    #aplico template de chat na entrada, e retorno valor em tokens
    inputs = tokenizer.apply_chat_template(
        historico_conversas, #carrego entradas, no caso é o padrão dos sistema  + historico + input do usuario
        add_generation_prompt=True,#adiciono no começo um token indicando que espero a respsota do modelo
        return_tensors="pt" #quando tokenizo entradas, essas entradas estão no formato de tensores do pytorch
        ).to(model.device)

    #solicito resposta, atenção, ela vem como sequência de tokens!
    outputs = model.generate(
        inputs,
        pad_token_id = tokenizer.eos_token_id,
        max_new_tokens=100, #limito a resposta a no máximo 100 tokens(padrão 256). Para ser consiso use etnre 50 e 100. Para criatividade, usae entre 200-500.
        do_sample=True, #deixe como False para pegar sempre o token mais provável, se true iremos definir como escolher o próximo token
        temperature=0.4, #padrão 0.7. use valor entre 0.1 e 0.5 para leitura de textos, entre 0.5 e 1 para uma conversa descontraida e entre 1.2 e 2 para brincar. Aqui mudamos a probabilidade de todos os tokens
        top_p=0.2 #padrão 0.9 escolhemos que o próximo token seja baseado nos 90% tokens que mais fazem sentido. Entre 0.1 e 0.5 focamos a precisão. Entre 0.9 e 1 liberamos a criatividade. Aqui selecionamos os tokens mais prováveis
    )
    
    #decodifico resposta, ou seja, transformo a sequência de valores de tokens, em uma sequência de strings
    resposta = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    #adiciono respsota ao historico
    historico_conversas.append({"role": "assistant", "content": resposta})
    return resposta
#####           PARTE CRIANDO INTERFACE
import chainlit as cl
@cl.on_chat_start
async def start_chat():
    await cl.Message(
        content="Ola, sou seu ChatBot que usa a inteligência do DeepSeek. Em que posso ajudar?"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    # Leio entrada do usuário
    entrada = message.content
    # Aviso que estou pensando
    await cl.Message(
        content=f"Entendi que você me mandou: {entrada} \n Estou pensando em como te responder..."
    ).send()
    # Busco resposta do chatbot a essa entrada
    resposta = await cl.make_async(PegoResposta)(entrada)
    # Envio resposta de volta
    await cl.Message(
        content=resposta
    ).send()
    
