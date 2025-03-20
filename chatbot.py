from langchain_ollama import OllamaLLM
from langchain_ollama.chat_models import ChatOllama

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Importação corrigida

import os

#  BASE DE CONHECIMENTO
def carregar_base_de_conhecimento(pasta="documentos"):
    textos = []
    for arquivo in os.listdir(pasta):
        if arquivo.endswith(".txt"):
            with open(os.path.join(pasta, arquivo), "r", encoding="utf-8") as f:
                textos.append(f.read())

    # Dividir os textos para melhor indexação
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    textos_divididos = splitter.split_text("\n".join(textos))  # Corrigido para trabalhar com texto puro

    # Criar os embeddings usando Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Criar a base de conhecimento com FAISS
    db = FAISS.from_texts(textos_divididos, embeddings)
    return db

base = carregar_base_de_conhecimento()


#  MEMÓRIA
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

memory = ConversationBufferMemory(memory_key="history")

template = PromptTemplate(
    input_variables=["history", "pergunta"],
    template=(
        "Você é um assistente virtual que responde **apenas em português**. "
        "Se a pergunta estiver em outro idioma, traduza e responda em português. "
        "Se não souber a resposta, diga 'Não sei'.\n\n"
        "{history}\nUsuário: {pergunta}\nAssistente:"
    )
)

# Criar o modelo de IA Mistral via Ollama
llm = OllamaLLM(model="mistral", temperature=0.5, system="Responda **somente** em português.")

# Criar a cadeia de resposta do chatbot
chatbot = LLMChain(llm=llm, prompt=template, memory=memory)


#  INFORMAÇÕES DA EMPRESA
def carregar_informacoes_arquivo(arquivo="informacoes.txt"):
    try:
        with open(arquivo, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return "Nenhuma informação disponível."

informacoes = carregar_informacoes_arquivo()

# Criar o prompt da IA com o conteúdo do arquivo
prompt = f"""
Você é um assistente virtual da empresa Nyxi Tecnologia. Seu objetivo é fornecer respostas precisas sobre a empresa, seus serviços e seus valores.  
Se a pergunta for sobre a empresa, utilize as informações disponíveis. Se não souber, diga 'Não sei' em vez de inventar uma resposta.  
Sempre responda **somente em português**.

Aqui estão as informações que você deve utilizar para responder:
{informacoes}

Responda **sempre** em português, independentemente do idioma da pergunta.  
Se a pergunta for sobre a empresa, use as informações fornecidas.  
Se não houver informação suficiente, diga 'Não sei'.
"""


#  EXECUTAR CHATBOT
def chatbot_interativo():
    print("Chatbot IA - Digite 'sair' para encerrar")
    while True:
        pergunta = input("Você: ")
        if pergunta.lower() == "sair":
            break
        resposta = llm.invoke(pergunta)
        print("Bot:", resposta)

chatbot_interativo()
