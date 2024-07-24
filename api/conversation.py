import os
import getpass
import json
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

llm = ChatGoogleGenerativeAI(model="gemini-pro")
google_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def supervisorAgent(query, llm):
  prompt_template = """
  Você é um profissional de RH e o seu trabalho é fazer a análise de currículos que serão enviados no corpo da mensagem no formato de um link para baixar um arquivo no formato DOCX ou PDF. Os currículos são para pessoas que desejam trabalhar na área de desenvolvimento de software. Com base no conteúdo do curriculo que vou te enviar no corpo da mensagem, o retorno que eu gostaria de receber é uma lista com as seguintes informações:
  - Nome do candidato.
  - Idade.
  - Email.
  - Link da rede Linkedin.
  - Link da rede Github.
  - Telefone para contato.
  - Possui ensino superior completo (sim/não).
  - Tecnologias que o candidato conhece.
  - Tempo de experiência profissional.
  - Tempo médio que o candidato permanece no emprego.
  - Se o candidato conhece alguma lingua estrangeira, liste o idioma que o candidato informou.
  - Se o perfil predominante do candidato é backend ou frontend.
  Pode ser que todas as informações que eu preciso não sejam localizadas no currículo enviado. Neste caso, basta retornar que não localizou a informação solicitada.
  Na sequência, liste as experiências profissionais que foram localizadas no currículo.
  Além desta lista, se o candidato enviou o link para a rede Linkedin, faça a seguinte análise, acesse o link informado e responda se existe alguma informação profissional no Linkedin que não tenha sido enviada no currículo.
  Se o link do Linkedin enviado começar com "www", acrescentar "https://" no começo antes de acessar.
  Além desta lista, se o candidato tiver enviado o link para a rede GitHub, faça a seguinte análise, verifique quantos repositórios existem, um resumo das tecnologias utilizadas e quando foi a última atualização feita.
  Você pode, ao final de todas estas informações, recomendar melhorias no currículo, caso você considere que faltem detalhes ou que ele seja pouco atraente.
  Usuário: {query}
  Assistente:
  """

  prompt = PromptTemplate(
    input_variables=['query'],
    template = prompt_template
  )

  sequence = RunnableSequence(prompt | llm)
  response = sequence.invoke({"query": query})
  return response

def getResponse(query, llm):
  response = supervisorAgent(query, llm)
  return response



def conversation(query):
  # query = event.get("question")
  response = getResponse(query, llm).content
  return response