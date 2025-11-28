from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

plantilla = PromptTemplate(
    input_variables=["nombre"],
    template="Saluda al usuario con su nombre.\nNombre del usuario: {nombre}\n:"
)

#Encadenamos operaciones con | (operador pipe o tubería)
chain = plantilla | chat

resultado = chain.invoke({"nombre": "Jorge"})
print(resultado.content)