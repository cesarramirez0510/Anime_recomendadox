from langchain_ollama.llms import OllamaLLM

llm = OllamaLLM(model="tinyllama")

pregunta = input("What can i do for you? ")
respuesta = llm.invoke(pregunta)
print("Answer:", respuesta.content)