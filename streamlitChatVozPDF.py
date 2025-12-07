import streamlit as st
import os
import io
import glob
from pydub import AudioSegment

# --- 1. CONFIGURACIÓN DE PÁGINA (Debe ser lo primero) ---
st.set_page_config(
    page_title="Tesis V1 - Chat PDF", 
    page_icon="🎙️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ESTILOS CSS PERSONALIZADOS (UI MEJORADA) ---
st.markdown("""
<style>
    /* Importar fuentes académicas */
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;700&family=Roboto:wght@300;400;500&display=swap');

            
            
    /* 1. Ocultar SOLO el menú de opciones (3 puntos) y el footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 2. AJUSTE CRÍTICO: NO ocultar el header completo, solo hacerlo discreto */
    /* Esto permite que el botón de la sidebar siga funcionando */
    header {
        visibility: visible !important;
        background-color: transparent !important;
    }
    
    /* Opcional: Si quieres ocultar la línea de colores de decoración superior */
    div[data-testid="stDecoration"] {
        visibility: hidden;
    }
    
    /* Tipografía Global */
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    /* Títulos Académicos */
    h1, h2, h3 {
        font-family: 'Merriweather', serif !important;
        color: #711610;
    }

    /* Contenedor Principal */
    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* BURBUJAS DE CHAT */
    .stChatMessage {
        background-color: transparent;
        border: none;
        padding: 1rem 0;
        border-bottom: 1px solid #e2e8f0;
    }

    /* Iconos de Avatar */
    .stChatMessage .st-emotion-cache-1p1m4ay {
        background-color: #0f172a;
        color: white;
    }

    /* Botones */
    .stButton button {
        background-color: #711610 !important;
        color: white !important;
        border-radius: 4px;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 1px;
        border: none;
        transition: all 0.2s;
    }
    .stButton button:hover {
        background-color: #AA5757 !important;
    }
    
    /* Citas */
    .citation-box {
        background-color: #f1f5f9;
        border-left: 3px solid #0f172a;
        padding: 10px;
        font-size: 0.9rem;
        font-family: 'Roboto', sans-serif;
        color: #475569;
        margin-top: 5px;
    }
            
            
    </style>
""", unsafe_allow_html=True)

# --- 3. CONFIGURACIÓN FFMPEG (Solución WinError 2) ---
try:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    path_ffmpeg = os.path.join(root_dir, "ffmpeg.exe")
    path_ffprobe = os.path.join(root_dir, "ffprobe.exe")
    
    AudioSegment.converter = path_ffmpeg
    AudioSegment.ffmpeg = path_ffmpeg
    AudioSegment.ffprobe = path_ffprobe
    
    os.environ["PATH"] += os.pathsep + root_dir
except Exception as e:
    st.error(f"⚠️ Error configurando audio local: {e}")

# --- 4. IMPORTACIONES DE LIBRERÍAS (Resto) ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from gtts import gTTS
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
# --- 4. HEADER INSTITUCIONAL ---
col_logo, col_title = st.columns([0.15, 0.85])
with col_logo:
    # Busca primero el SVG (Prioridad 1)
    if os.path.exists("logo.svg"):
        st.image("logo.svg", width=120) 
    # Si no, busca el PNG (Prioridad 2)
    elif os.path.exists("logo.png"):
        st.image("logo.png", width=120)
    # Si no hay nada, usa el Emoji (Respaldo)
    else:
        st.markdown("<div style='font-size:3rem; text-align:center;'>🏛️</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("# CHATBOY RAG - GRUPO 03")
    # ---st.markdown("**PhD Candidate:** [Tu Nombre] | **Advisor:** [Nombre Asesor]")---
    st.markdown("---")

# --- CONFIGURACIÓN API ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.error("⚠️ Falta la API Key en .streamlit/secrets.toml")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=GOOGLE_API_KEY
)

# --- FUNCIONES AUXILIARES ---

def texto_a_audio(texto):
    try:
        tts = gTTS(text=texto, lang='es')
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        return None

def transcribir_audio(audio_bytes):
    r = sr.Recognizer()
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        with sr.AudioFile(wav_io) as source:
            audio_data = r.record(source)
            return r.recognize_google(audio_data, language="es-ES")
    except sr.UnknownValueError:
        return None
    except Exception as e:
        return f"Error: {e}"

@st.cache_resource
def cargar_vectorstore(archivo_pdf):
    # Usamos cache para no reprocesar el PDF cada vez que hablas
    rutaIndice = archivo_pdf.replace(".pdf", "")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    if os.path.exists(rutaIndice):
        try:
            return FAISS.load_local(rutaIndice, embeddings, allow_dangerous_deserialization=True).as_retriever()
        except:
            pass
            
    loader = PyPDFLoader(file_path=archivo_pdf)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50, separator="\n")
    docs = text_splitter.split_documents(documents=documents)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(rutaIndice)
    return vectorstore.as_retriever()

def generar_respuesta(pregunta, retriever):
    system_prompt = (
        "Eres un experto en el documento proporcionado. "
        "Responde de manera concisa, natural y hablada (máximo 3 frases). "
        "Si no está en el texto, dilo amablemente. Contexto: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
    return chain.invoke({"input": pregunta})

# --- LÓGICA DE ESTADO (SESSION STATE) ---
if "messages" not in st.session_state: st.session_state.messages = []
if "archivo_actual" not in st.session_state: st.session_state.archivo_actual = ""

# --- BARRA LATERAL (SIDEBAR) MEJORADA ---
with st.sidebar:
    st.header("📂 Documentos")
    archivos = glob.glob("*.pdf")
    
    if archivos:
        seleccion = st.selectbox("Selecciona PDF:", archivos)
        
        # Botón para limpiar chat
        st.markdown("---")
        col_b1, col_b2 = st.columns(2)
        if col_b1.button("🗑️ Limpiar", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        if col_b2.button("🔄 Recargar", use_container_width=True):
            st.rerun()

        # Resetear historial si cambia el archivo
        if st.session_state.archivo_actual != seleccion:
            st.session_state.archivo_actual = seleccion
            st.session_state.messages = []
            st.rerun()
    else:
        st.warning("No hay PDFs en la carpeta.")
        st.stop()

# Cargar BD
retriever = cargar_vectorstore(seleccion)

# --- MOSTRAR HISTORIAL DE CHAT ---
avatares = {"user": "👤", "assistant": "🤖"}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=avatares[msg["role"]]):
        st.markdown(msg["content"])

# --- ÁREA DE INPUT (TEXTO + VOZ) ---
st.markdown("---")
col_txt, col_mic = st.columns([0.85, 0.15], gap="small")

with col_txt:
    texto_input = st.chat_input("Escribe tu pregunta aquí...")

with col_mic:
    st.write("") # Espaciador visual
    audio_data = mic_recorder(start_prompt="🎙️", stop_prompt="⏹️", key='recorder', format="webm")

# --- PROCESAMIENTO ---
prompt_final = None

if texto_input:
    prompt_final = texto_input
elif audio_data:
    st.toast("Procesando audio...") # Notificación flotante elegante
    transcripcion = transcribir_audio(audio_data['bytes'])
    if transcripcion and "Error" not in transcripcion:
        prompt_final = transcripcion
    elif transcripcion:
        st.error(transcripcion)
    else:
        st.warning("No se entendió el audio.")

if prompt_final:
    # 1. Guardar y mostrar usuario
    st.session_state.messages.append({"role": "user", "content": prompt_final})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_final)

    # 2. Generar respuesta
    try:
        with st.spinner("Pensando..."):
            res = generar_respuesta(prompt_final, retriever)
            respuesta_texto = res["answer"]
            
        # 3. Guardar y mostrar IA
        st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})
        
        with st.chat_message("assistant", avatar="🤖"):
            st.write(respuesta_texto)
            
            # Audio automático
            audio_bytes = texto_a_audio(respuesta_texto)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            
            # Referencias desplegables
            with st.expander("📚 Ver referencia en el PDF"):
                for doc in res["context"]:
                    st.info(doc.page_content[:250] + "...")

    except Exception as e:
        st.error(f"Error en la respuesta: {e}")
