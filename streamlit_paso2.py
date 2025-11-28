import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage

# Configuración de la página
st.set_page_config(
    page_title="Causatronimus AI", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para colores y tema
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .welcome-msg {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar con opciones
with st.sidebar:
    st.markdown("### 🎛️ Panel de Control de Causatronimus")
    
    # Configuración del modelo IA
    st.markdown("#### 🧠 Configuración del Modelo")
    
    # Selector de modelo
    modelo_seleccionado = st.selectbox(
        "Modelo de IA:",
        [
            "gemini-pro",
            "gemini-1.5-flash", 
            "gemini-1.5-pro"
        ],
        index=0,  # Empezar con gemini-pro
        key="modelo_ia",
        help="Selecciona el modelo de IA que quieres usar"
    )
    
    # Slider para temperatura
    temperatura = st.slider(
        "Temperatura (Creatividad):",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="0.0 = Muy conservador, 1.0 = Muy creativo",
        key="temperatura"
    )
    
    # Información del modelo actual
    st.info(f"🤖 **Modelo activo:** {modelo_seleccionado}\n\n🌡️ **Temperatura:** {temperatura}")
    
    st.divider()
    
    # Opciones de personalización
    st.markdown("#### 🎨 Personalización")
    user_name = st.text_input("¿Cómo te llamas, humano?", value="Usuario", key="user_name")
    
    # Selección de avatares
    st.markdown("#### 👤 Avatares")
    bot_avatar = st.selectbox("Avatar de Causatronimus:", ["🤖", "🦾", "👾", "🧠", "⚡", "🔮"], key="bot_avatar")
    user_avatar = st.selectbox("Tu Avatar:", ["😊", "👨‍💻", "👩‍💻", "🧑‍🎓", "👤", "🌟"], key="user_avatar")
    
    st.divider()
    
    # Estadísticas del chat
    st.markdown("#### 📊 Estadísticas")
    if "mensajes" in st.session_state:
        total_msgs = len(st.session_state.mensajes)
        user_msgs = len([msg for msg in st.session_state.mensajes if isinstance(msg, HumanMessage)])
        bot_msgs = len([msg for msg in st.session_state.mensajes if isinstance(msg, AIMessage)])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Tus mensajes", user_msgs)
        with col2:
            st.metric("Respuestas", bot_msgs)
        
        st.metric("Total conversación", total_msgs)
    
    st.divider()
    
    # Botón para limpiar chat
    if st.button("🗑️ Limpiar Chat", use_container_width=True):
        st.session_state.mensajes = []
        st.rerun()

# Header principal con gradiente
st.markdown(f"""
<div class="main-header">
    <h1 style="color: white; margin: 0; text-align: center;">
        🤖 Bienvenido humano, soy Causatronimus - Tu chatbot de confianza
    </h1>
</div>
""", unsafe_allow_html=True)

# Mensaje de bienvenida atractivo
if "mensajes" not in st.session_state or len(st.session_state.mensajes) == 0:
    st.markdown(f"""
    <div class="welcome-msg">
        <h2>¡Saludos {user_name if 'user_name' in st.session_state else 'humano'}! 👋</h2>
        <p>Soy <strong>Causatronimus</strong>, tu asistente robótico súper inteligente.</p>
        <p>🚀 Estoy en desarrollo pero ya puedo conversar contigo y ayudarte.</p>
        <p>💬 ¡Escribe algo abajo y empecemos a chatear!</p>
        <p><em>Construido con LangChain + Streamlit :D</em></p>
    </div>
    """, unsafe_allow_html=True)

# Configurar el modelo dinámicamente según el sidebar
try:
    chat_model = ChatGoogleGenerativeAI(
        model=st.session_state.get("modelo_ia", "gemini-2.0-flash-exp"),
        google_api_key="AIzaSyCzx74nncs7Qs5ITaNOVh9pcDv-mhn8DZM",
        temperature=st.session_state.get("temperatura", 0.7)
    )
except Exception as e:
    st.error(f"❌ Error configurando modelo: Posible limite de cuota alcanzado. Intenta en unos minutos.")
    st.info("💡 Tip: El API gratuito tiene límites de uso por minuto")
    st.stop()

# Inicializar el historial de mensajes en session_state
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []


# Renderizar historial existente
for msg in st.session_state.mensajes:
    if isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar=st.session_state.get("bot_avatar", "🤖")):
            st.markdown(msg.content)
    else:
        with st.chat_message("user", avatar=st.session_state.get("user_avatar", "😊")):
            st.markdown(msg.content)

# Input de usuario
pregunta = st.chat_input("Escribe tu mensaje:")

if pregunta:
    # Mostrar y almacenar mensaje del usuario
    with st.chat_message("user", avatar=st.session_state.get("user_avatar", "😊")):
        st.markdown(pregunta)
    
    st.session_state.mensajes.append(HumanMessage(content=pregunta))

    # Mostrar indicador de que Causatronimus está pensando
    with st.spinner('🤖 Causatronimus está procesando...'):
        try:
            respuesta = chat_model.invoke(st.session_state.mensajes)
            
            with st.chat_message("assistant", avatar=st.session_state.get("bot_avatar", "🤖")):
                st.markdown(respuesta.content)
                
            # Almacenar respuesta del bot
            st.session_state.mensajes.append(AIMessage(content=respuesta.content))
            
        except Exception as e:
            with st.chat_message("assistant", avatar=st.session_state.get("bot_avatar", "🤖")):
                st.error("⚠️ **Causatronimus dice:** ¡Ups! Parece que he alcanzado mi límite de energía por hoy.")
                st.info("💡 **Consejo:** El API gratuito tiene límites. Intenta en unos minutos y funcionaré perfectamente.")
                st.markdown("🔄 **Mientras tanto:** Puedes seguir escribiendo y tus mensajes se guardarán para cuando vuelva.")
    
    # Efecto de celebración ocasional
    import random
    if random.choice([True, False]):
        st.balloons()