import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

# Configuración inicial
st.set_page_config(page_title="Chatbot Básico", page_icon="🤖")
st.title("🤖 Chatbot - paso 1 - con LangChain")
st.markdown("Este es un *chatbot de ejemplo* construido con LangChain + Streamlit.")

# Modelo con API Key y modelo correcto
try:
    chat_model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp", 
        google_api_key="AIzaSyCzx74nncs7Qs5ITaNOVh9pcDv-mhn8DZM"
    )
except Exception as e:
    st.error(f" Error: {e}")
    st.info("Posible límite de cuota alcanzado")
    st.stop()

pregunta = st.chat_input("Escribe tu mensaje:")

if pregunta:
    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(pregunta)
    
    # Procesar y mostrar respuesta con manejo de errores
    try:
        respuesta = chat_model.invoke(pregunta)
        with st.chat_message("assistant"):
            st.markdown(respuesta.content)
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(" Error de cuota del API. Intenta en unos minutos.")
            st.info(" El API gratuito tiene límites por minuto")