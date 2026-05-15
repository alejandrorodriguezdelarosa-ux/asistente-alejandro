"""
Asistente Personal de Alejandro Rodríguez de la Rosa
====================================================

Chatbot RAG que responde preguntas sobre Alejandro a reclutadores
y personas interesadas en su perfil profesional.

Stack: OpenAI (gpt-4o-mini + text-embedding-3-small) + ChromaDB + LangGraph + Streamlit.

Ejecución local:
    streamlit run streamlit_app.py

Requiere la variable de entorno OPENAI_API_KEY (en .env o, en Streamlit Cloud,
en Settings → Secrets).
"""

import os
import re
import uuid
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv, find_dotenv

# Cargar .env desde la carpeta del proyecto (busca recursivamente hacia arriba).
load_dotenv(find_dotenv(usecwd=True))


# ─────────────────────────────────────────────────────────────────────────────
# SUPABASE — Guardado de preguntas (no bloqueante, falla silenciosa)
# ─────────────────────────────────────────────────────────────────────────────

def _get_supabase_config():
    """Lee la URL y la publishable key de Supabase desde variables de entorno
    o desde los secrets de Streamlit Cloud."""
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        try:
            url = url or st.secrets.get("SUPABASE_URL", "")
            key = key or st.secrets.get("SUPABASE_KEY", "")
        except (FileNotFoundError, KeyError):
            pass
    return url, key


def guardar_pregunta(pregunta: str) -> None:
    """Guarda una pregunta en la tabla `preguntas` de Supabase.

    VERSIÓN CON DIAGNÓSTICO TEMPORAL: muestra el resultado en la barra lateral
    para poder identificar qué falla. Una vez funcione, volvemos a la versión
    silenciosa.
    """
    url, key = _get_supabase_config()

    # Diagnóstico paso 1: ¿se leen los secrets?
    if not url or not key:
        st.sidebar.error(
            f"🔴 DEBUG: Secrets no encontrados. "
            f"URL={'OK' if url else 'VACÍO'}, KEY={'OK' if key else 'VACÍO'}"
        )
        return

    endpoint = f"{url.rstrip('/')}/rest/v1/preguntas"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    payload = {"pregunta": pregunta}

    # Diagnóstico paso 2: hacer la petición y mostrar el resultado
    try:
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=5)
        if resp.status_code in (200, 201, 204):
            st.sidebar.success(f"🟢 DEBUG: Guardado OK (status {resp.status_code})")
        else:
            st.sidebar.error(
                f"🔴 DEBUG: Supabase devolvió {resp.status_code}. "
                f"Respuesta: {resp.text[:300]}"
            )
    except Exception as e:
        st.sidebar.error(f"🔴 DEBUG: Excepción al llamar a Supabase: {type(e).__name__}: {e}")



# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Asistente de Alejandro Rodríguez",
    page_icon="🧑‍💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }

    .stat-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #3d4166;
        border-radius: 12px;
        padding: 14px 10px;
        text-align: center;
        margin: 4px 0;
    }
    .stat-number { font-size: 1.8rem; font-weight: bold; color: #4ade80; }
    .stat-label  { color: #8b8fa8; font-size: 0.8rem; margin-top: 2px; }

    .rag-panel {
        background: #0f1923;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 12px;
        font-size: 0.82rem;
        color: #7dd3fc;
        font-family: monospace;
        max-height: 200px;
        overflow-y: auto;
        margin-top: 8px;
    }

    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ESTADO DE SESIÓN
# ─────────────────────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "messages": [],          # Historial visible en la UI
        "total_calls": 0,
        "thread_id": f"streamlit-{uuid.uuid4().hex[:8]}",
        "agente": None,
        "retriever": None,
        "ultimo_contexto_rag": "",
        "api_key_ok": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — chatbot personal de Alejandro para reclutadores
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres el asistente personal de Alejandro Rodríguez de la Rosa, integrado en su portafolio profesional.
Hablas en su nombre con reclutadores, headhunters y personas interesadas en su perfil profesional.

INSTRUCCIONES:
1. Basa SIEMPRE tus respuestas EXCLUSIVAMENTE en el contexto recuperado de la base de conocimiento. NO inventes datos sobre Alejandro. Si algo no está en el contexto, responde EXACTAMENTE con esta frase, sin añadir nada más: "Si deseas conocer esa información, ¡deberías hacerle una entrevista!"
2. Responde SIEMPRE en español, en tercera persona ("Alejandro ha trabajado…", "su experiencia es…"), nunca en primera persona como si fueras él.
3. Sé objetivo y honesto: no exageres ni adornes las respuestas. Si Alejandro reconoce un defecto o un fracaso, no lo escondas, contextualízalo.
4. Mantén un tono profesional pero cercano, como respondería un asistente bien entrenado de un candidato serio.
5. Estructura tus respuestas de forma clara pero breve: los reclutadores no quieren párrafos largos. Usa listas solo cuando aporten claridad.
6. Si te preguntan por el salario o expectativas económicas, responde EXACTAMENTE: "Si deseas conocer esa información, ¡deberías hacerle una entrevista!"
7. Si te preguntan algo personal incómodo, fuera de lugar o no relacionado con su perfil profesional o personal documentado, responde EXACTAMENTE: "Si deseas conocer esa información, ¡deberías hacerle una entrevista!"
8. Si el usuario pregunta cómo contactar con Alejandro, proporciónale el correo, teléfono o LinkedIn que aparezcan en la base de conocimiento.
9. Aprovecha la memoria de la conversación: si el usuario ya ha preguntado algo, conecta tus respuestas con el contexto previo.
10. Al final de respuestas largas o cuando sea natural hacerlo, sugiere de forma breve una pregunta de seguimiento útil para un reclutador (ejemplo: "¿Quieres que te cuente más sobre su experiencia en GEO?")."""


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING POR BLOQUES ID (en lugar de RecursiveCharacterTextSplitter)
# ─────────────────────────────────────────────────────────────────────────────

def cargar_bloques_rag(ruta_md: str):
    """Lee el .md de Alejandro y devuelve una lista de bloques autocontenidos.

    Cada bloque del archivo tiene el formato:
        ---
        ID: X.Y
        HECHO: ...
        ENTIDADES: ...
        PALABRAS_CLAVE: ...
        ---

    En lugar de cortar por tamaño fijo, se trata cada bloque como una unidad
    indivisible: así el RAG recupera hechos completos en vez de fragmentos.
    """
    from langchain_core.documents import Document

    with open(ruta_md, "r", encoding="utf-8") as f:
        texto = f.read()

    # Patrón: capturamos bloques que contengan "ID: X.Y" y los tres campos siguientes.
    patron = re.compile(
        r"ID:\s*(?P<id>[\d\.]+)\s*\n"
        r"HECHO:\s*(?P<hecho>.+?)\s*\n"
        r"ENTIDADES:\s*(?P<entidades>.+?)\s*\n"
        r"PALABRAS_CLAVE:\s*(?P<keywords>.+?)\s*(?=\n---)",
        re.DOTALL,
    )

    documentos = []
    for m in patron.finditer(texto):
        bloque_id = m.group("id").strip()
        hecho = m.group("hecho").strip()
        entidades = m.group("entidades").strip()
        keywords = m.group("keywords").strip()

        # Contenido enriquecido para el embedding: incluimos keywords para
        # mejorar la recuperación por términos relacionados.
        contenido = (
            f"{hecho}\n\n"
            f"Entidades: {entidades}\n"
            f"Palabras clave: {keywords}"
        )

        documentos.append(Document(
            page_content=contenido,
            metadata={
                "id_bloque": bloque_id,
                "source": "alejandro_rag_completo.md",
            },
        ))

    return documentos


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DEL AGENTE (cacheada para no recompilar en cada rerun de Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def cargar_agente(api_key: str, modelo: str):
    """Construye el vectorstore, el LLM y el grafo LangGraph.

    Retorna: (agente_compilado, retriever, n_bloques_indexados).
    """
    from typing import Annotated, TypedDict

    from langchain_community.vectorstores import Chroma
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    DOCS_DIR = "docs"
    CHROMA_DIR = "chroma_db"
    RUTA_MD = os.path.join(DOCS_DIR, "alejandro_rag_completo.md")

    if not os.path.exists(RUTA_MD):
        raise FileNotFoundError(
            f"No se encontró el documento de conocimiento: {RUTA_MD}. "
            "Debe existir en la carpeta docs/."
        )

    # 1) Carga + chunking por bloques ID
    documentos = cargar_bloques_rag(RUTA_MD)

    if not documentos:
        raise ValueError(
            "No se pudo extraer ningún bloque del archivo Markdown. "
            "Comprueba que el formato sigue siendo ID/HECHO/ENTIDADES/PALABRAS_CLAVE."
        )

    # 2) Embeddings + ChromaDB persistente (colección nueva)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )
    vectorstore = Chroma.from_documents(
        documents=documentos,
        embedding=embeddings,
        collection_name="alejandro_perfil",
        persist_directory=CHROMA_DIR,
    )
    # k=5 porque los bloques son más cortos que los chunks de PDF originales;
    # recuperar más bloques compensa esa granularidad fina.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 3) LLM — temperatura 0.2 como pediste en el brief inicial
    llm = ChatOpenAI(
        model=modelo,
        temperature=0.2,
        openai_api_key=api_key,
    )

    # 4) Estado del grafo
    class EstadoAsistente(TypedDict):
        mensajes: Annotated[list[BaseMessage], add_messages]
        contexto_rag: str

    # Nodo 1: recuperación RAG
    def nodo_rag(estado):
        ultimo = None
        for msg in reversed(estado["mensajes"]):
            if isinstance(msg, HumanMessage):
                ultimo = msg.content
                break
        if not ultimo:
            return {"contexto_rag": ""}
        docs = retriever.invoke(ultimo)
        if not docs:
            return {"contexto_rag": "No se encontró información en la base de conocimiento."}
        fragmentos = []
        for i, doc in enumerate(docs, 1):
            id_bloque = doc.metadata.get("id_bloque", "?")
            fragmentos.append(f"[Bloque {id_bloque}]\n{doc.page_content}")
        return {"contexto_rag": "\n\n".join(fragmentos)}

    # Nodo 2: generación con OpenAI
    def nodo_generacion(estado):
        contexto = estado.get("contexto_rag", "")
        sys_prompt = SYSTEM_PROMPT
        if contexto:
            sys_prompt += (
                f"\n\nCONTEXTO DE LA BASE DE CONOCIMIENTO:\n{'=' * 50}\n"
                f"{contexto}\n{'=' * 50}\n"
                "Usa este contexto como ÚNICA fuente de información sobre Alejandro."
            )
        mensajes_completos = [SystemMessage(content=sys_prompt)] + estado["mensajes"]
        respuesta = llm.invoke(mensajes_completos)
        return {"mensajes": [respuesta]}

    grafo = StateGraph(EstadoAsistente)
    grafo.add_node("recuperar", nodo_rag)
    grafo.add_node("generar", nodo_generacion)
    grafo.add_edge(START, "recuperar")
    grafo.add_edge("recuperar", "generar")
    grafo.add_edge("generar", END)

    agente = grafo.compile(checkpointer=MemorySaver())
    return agente, retriever, len(documentos)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuración")
    st.divider()

    # API Key — admite valor del .env / secrets como pre-relleno
    api_key_input = st.text_input(
        "🔑 OpenAI API Key",
        type="password",
        placeholder="sk-...",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Necesaria solo si no está configurada en .env o en los secrets de Streamlit Cloud.",
    )
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input

    st.divider()
    st.subheader("🤖 Modelo")
    modelo = st.selectbox(
        "Modelo OpenAI",
        [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4.1-mini",
        ],
        index=0,
        help="gpt-4o-mini: rápido y económico (recomendado) · gpt-4o: máxima calidad",
    )

    st.divider()
    st.subheader("📊 Sesión")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"<div class='stat-card'>"
            f"<div class='stat-number'>{st.session_state.total_calls}</div>"
            f"<div class='stat-label'>Consultas</div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='stat-card'>"
            f"<div class='stat-number'>{len(st.session_state.messages)}</div>"
            f"<div class='stat-label'>Mensajes</div></div>",
            unsafe_allow_html=True,
        )

    st.caption(f"🔑 Thread: `{st.session_state.thread_id[-8:]}`")

    st.divider()
    if st.button("🗑️ Nueva conversación", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.total_calls = 0
        st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex[:8]}"
        st.session_state.ultimo_contexto_rag = ""
        st.rerun()

    st.divider()
    st.subheader("💡 Preguntas de ejemplo")
    ejemplos = [
        "¿Cuál es su formación académica?",
        "¿Qué experiencia tiene con IA y automatizaciones?",
        "¿Dónde se ve en 5 años?",
        "¿Por qué deberíamos contratarle?",
        "¿Cuál es su disponibilidad geográfica?",
    ]
    for ejemplo in ejemplos:
        if st.button(f"→ {ejemplo}", use_container_width=True, key=f"ej_{ejemplo}"):
            st.session_state["_pregunta_rapida"] = ejemplo
            st.rerun()

    with st.expander("🔍 Último contexto RAG"):
        if st.session_state.ultimo_contexto_rag:
            st.markdown(
                f"<div class='rag-panel'>{st.session_state.ultimo_contexto_rag[:800]}...</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Sin consultas aún.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Cabecera
# ─────────────────────────────────────────────────────────────────────────────

st.title("🧑‍💼 Asistente personal de Alejandro Rodríguez")
st.caption(
    "Chatbot RAG · OpenAI + ChromaDB + LangGraph  |  "
    "Pregúntame sobre la formación, experiencia y perfil profesional de Alejandro."
)


# ─────────────────────────────────────────────────────────────────────────────
# INICIALIZACIÓN DEL AGENTE
# ─────────────────────────────────────────────────────────────────────────────

api_key = os.getenv("OPENAI_API_KEY", "")

if not api_key:
    st.warning(
        "⚠️ Configura tu **OpenAI API Key** en el panel lateral para empezar. "
        "Puedes obtener una clave en "
        "[platform.openai.com](https://platform.openai.com/api-keys)."
    )
    st.stop()

try:
    with st.spinner("⚙️ Inicializando asistente y base de conocimiento..."):
        agente, retriever, n_bloques = cargar_agente(api_key, modelo)
    st.session_state.agente = agente
    st.session_state.retriever = retriever
    st.session_state.api_key_ok = True
except FileNotFoundError as e:
    st.error(f"❌ {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Error inicializando el agente: {e}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# HISTORIAL DE CHAT
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 **Hola, soy el asistente personal de Alejandro Rodríguez de la Rosa.**\n\n"
            "Estoy aquí para responder cualquier duda que tengas sobre su perfil profesional. "
            "Puedo contarte sobre:\n"
            "- 🎓 **Formación**: marketing, Data Science e IA, idiomas.\n"
            "- 💼 **Experiencia**: emprendimiento, automatizaciones, GEO, marketing.\n"
            "- 🛠️ **Habilidades técnicas**: Python, SQL, Power BI, n8n, IA generativa.\n"
            "- 🎯 **Objetivos y disponibilidad**: ubicación, modalidad de trabajo, expectativas.\n"
            "- 🧠 **Personalidad y valores**: cómo trabaja, qué le motiva, cómo encaja en un equipo.\n\n"
            "¿Qué quieres saber?"
        )
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tiempo"):
                st.caption(f"⏱️ {msg['tiempo']}ms · 🤖 {modelo}")


# ─────────────────────────────────────────────────────────────────────────────
# INPUT Y PROCESAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

pregunta_rapida = st.session_state.pop("_pregunta_rapida", None)
prompt = st.chat_input("Pregúntame algo sobre Alejandro...") or pregunta_rapida

if prompt:
    from langchain_core.messages import HumanMessage as HMsg

    # Guardado asíncrono y tolerante a fallos en Supabase (no bloquea la UI)
    guardar_pregunta(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Consultando la base de conocimiento..."):
            try:
                inicio = datetime.now()

                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                entrada = {"mensajes": [HMsg(content=prompt)]}
                resultado = agente.invoke(entrada, config=config)

                tiempo_ms = int((datetime.now() - inicio).total_seconds() * 1000)
                respuesta = resultado["mensajes"][-1].content
                contexto = resultado.get("contexto_rag", "")

                st.session_state.ultimo_contexto_rag = contexto

                st.markdown(respuesta)
                st.caption(
                    f"⏱️ {tiempo_ms}ms · 🤖 {modelo} · 📄 {n_bloques} bloques indexados"
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": respuesta,
                    "tiempo": tiempo_ms,
                })
                st.session_state.total_calls += 1

            except Exception as e:
                st.error(f"❌ Error al generar respuesta: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# AVISO DE PRIVACIDAD (discreto, en el footer)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    "<div style='text-align:center;color:#6b6f8a;font-size:0.72rem;"
    "margin-top:24px;padding:8px;opacity:0.7;'>"
    "Las consultas pueden almacenarse de forma anónima para mejorar el asistente."
    "</div>",
    unsafe_allow_html=True,
)
