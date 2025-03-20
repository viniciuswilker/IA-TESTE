from langchain_ollama import OllamaLLM
from langchain_ollama.chat_models import ChatOllama



# Ler o arquivo
with open("informacoes.txt", "r", encoding="utf-8") as file:
    informacoes = file.read()

# Exibir informações carregadas antes de configurar a IA
print("\nInformações carregadas para a IA:\n", informacoes)

# Criar o prompt da IA com o conteúdo do arquivo
prompt = """
Você é um assistente virtual da empresa Nyxi Tecnologia. Seu objetivo é fornecer respostas precisas sobre a empresa, seus serviços e seus valores.  
Se a pergunta for sobre a empresa, utilize as informações disponíveis. Se não souber, diga 'Não sei' em vez de inventar uma resposta.  
Sempre responda em português de forma clara e objetiva.
Aqui estão as informações que você deve utilizar para responder:
{informacoes}
Responda **sempre** em português, independentemente do idioma da pergunta.  
Se a pergunta for sobre a empresa, use as informações fornecidas.  
Se não houver informação suficiente, diga 'Não sei'.
"""


llm = OllamaLLM(model="mistral", system=prompt, temperature=0.5)

def chatbot():
    print("Chatbot IA - Digite 'sair' para encerrar")
    while True:
        pergunta = input("Você: ")
        if pergunta.lower() == "sair":
            break
        resposta = llm.invoke(pergunta)
        print("Bot:", resposta)

chatbot()


# Instalar o Ollama
# iwr -useb https://ollama.com/install.ps1 | iex

# Baixar o Modelo Mistral
# ollama pull mistral


# Instalar o LangChain e Dependências
# pip install langchain langchain_ollama faiss-cpu
# pip install -U langchain langchain-community sentence-transformers
