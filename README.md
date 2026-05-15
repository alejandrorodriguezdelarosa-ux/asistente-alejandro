# Asistente Personal de Alejandro Rodríguez — Chatbot RAG

Chatbot RAG que responde preguntas sobre el perfil profesional de Alejandro Rodríguez de la Rosa, diseñado para integrarse en su portafolio personal y servir como punto de entrada para reclutadores.

---

## 1. Qué hace

Es un asistente conversacional que recupera información de una base de conocimiento personal (un Markdown estructurado con 107 bloques sobre formación, experiencia, habilidades, personalidad, disponibilidad y proyectos) y responde preguntas de reclutadores en lenguaje natural, manteniendo memoria entre turnos.

Casos de uso típicos:

- "¿Qué experiencia tiene con IA?"
- "¿Cuál es su disponibilidad para mudarse?"
- "¿Por qué cerró la asesoría GEO?"
- "¿En qué proyectos ha trabajado?"
- "¿Cómo lo contacto?"

---

## 2. Stack tecnológico

| Componente | Tecnología |
|---|---|
| LLM | OpenAI `gpt-4o-mini` (configurable a `gpt-4o` o `gpt-4.1-mini`) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Base vectorial | ChromaDB con persistencia en disco |
| Framework de agente | LangGraph (sobre LangChain) |
| Memoria | `MemorySaver` de LangGraph (por `thread_id`) |
| Interfaz | Streamlit |
| Despliegue | Streamlit Community Cloud |

---

## 3. Estructura del proyecto

```
.
├── streamlit_app.py                Aplicación principal
├── requirements.txt                Dependencias
├── .env.example                    Plantilla de variables de entorno
├── .gitignore
├── README.md
└── docs/
    └── alejandro_rag_completo.md   Base de conocimiento (107 bloques)
```

Nota: la carpeta `chroma_db/` se crea automáticamente en la primera ejecución y está excluida del repositorio.

---

## 4. Diferencias clave respecto a un RAG estándar

Tres decisiones técnicas adaptadas al caso concreto:

1. **Chunking por bloque ID en lugar de por tamaño fijo.** El Markdown fuente ya tiene bloques autocontenidos (`ID / HECHO / ENTIDADES / PALABRAS_CLAVE`). Cortar por tamaño rompería estos bloques; mantenerlos como unidad indivisible mejora la precisión de la recuperación. Cada chunk = un hecho completo.
2. **Embedding enriquecido con palabras clave.** El contenido vectorizado incluye el hecho + las palabras clave, lo que mejora la recuperación cuando el reclutador usa sinónimos o términos relacionados.
3. **Top-k = 5 en lugar de 3.** Los bloques son más cortos que los chunks de PDFs típicos, así que recuperar más bloques compensa la granularidad fina sin saturar el contexto.

---

## 5. Ejecución local

### Requisitos

- Python 3.10 o superior.
- Una API key de OpenAI ([platform.openai.com](https://platform.openai.com/api-keys)).

### Pasos

1. Clonar el repositorio y entrar en la carpeta:

   ```bash
   git clone https://github.com/TU_USUARIO/TU_REPO.git
   cd TU_REPO
   ```

2. Crear un entorno virtual e instalar dependencias:

   ```bash
   python -m venv venv
   source venv/bin/activate          # Linux / Mac
   venv\Scripts\activate             # Windows
   pip install -r requirements.txt
   ```

3. Configurar la API key:

   ```bash
   cp .env.example .env
   ```

   Editar `.env` y poner la clave real:

   ```
   OPENAI_API_KEY=sk-...
   ```

4. Lanzar la app:

   ```bash
   streamlit run streamlit_app.py
   ```

   Se abrirá en `http://localhost:8501`.

---

## 6. Despliegue en Streamlit Community Cloud

### Paso a paso

1. **Subir el proyecto a GitHub** (público o privado, ambos funcionan):

   ```bash
   git add .
   git commit -m "Asistente RAG personal"
   git push origin main
   ```

   Confirma que `.env` NO se ha subido (debe estar excluido por `.gitignore`).

2. **Ir a Streamlit Cloud:** [share.streamlit.io](https://share.streamlit.io) y conectar tu cuenta de GitHub.

3. **Crear nueva app:**
   - Repository: el repositorio del proyecto.
   - Branch: `main`.
   - Main file path: `streamlit_app.py`.

4. **Configurar el secret de la API key.** Antes de desplegar, hacer clic en *Advanced settings → Secrets* y pegar:

   ```toml
   OPENAI_API_KEY = "sk-tu_clave_real"
   ```

5. **Pulsar Deploy.** En 1-2 minutos tendrás una URL pública del tipo `https://tunombre-asistente.streamlit.app`.

### Notas

- La primera ejecución tarda más porque indexa la base en ChromaDB. Las siguientes son inmediatas.
- Cada vez que hagas `git push` a `main`, Streamlit Cloud redepliega automáticamente.

---

## 7. Integración en el portafolio

Hay dos formas de enlazar el chatbot desde portafolio.óptimoia.es. La elección depende de si tu constructor de portafolio permite insertar HTML personalizado.

### Opción A — Botón con enlace directo (más simple, siempre funciona)

Añade un botón en tu portafolio que abra el chatbot en una nueva pestaña:

```html
<a href="https://tunombre-asistente.streamlit.app" target="_blank" rel="noopener">
  💬 Habla con mi asistente
</a>
```

### Opción B — Iframe embebido (mejor experiencia, requiere HTML personalizado)

Si tu constructor de portafolio permite insertar HTML, embebe el chat en una sección:

```html
<iframe
  src="https://tunombre-asistente.streamlit.app?embed=true"
  width="100%"
  height="700"
  frameborder="0"
  style="border-radius: 12px;"
></iframe>
```

El parámetro `?embed=true` oculta el menú superior y el footer de Streamlit para que parezca una sección nativa del portafolio.

---

## 8. Costes estimados

Con `gpt-4o-mini` y `text-embedding-3-small`:

- **Indexado inicial** (107 bloques, una sola vez): ~$0.001 (un céntimo).
- **Consulta media** (pregunta + 5 bloques recuperados + respuesta): ~$0.0005 - $0.002.

Con 1.000 consultas/mes, el coste total ronda los **$0.5 - $2/mes**. Cifras orientativas, los precios oficiales actualizados están en [openai.com/pricing](https://openai.com/pricing).

---

## 9. Actualizar la base de conocimiento

Para añadir, modificar o eliminar información sobre Alejandro:

1. Editar el archivo `docs/alejandro_rag_completo.md` manteniendo el formato exacto de bloques:

   ```
   ---
   ID: X.Y
   HECHO: ...
   ENTIDADES: ...
   PALABRAS_CLAVE: ...
   ---
   ```

2. Eliminar la carpeta `chroma_db/` (local) o forzar reindexado en Streamlit Cloud (reboot de la app desde el panel de control).

3. Hacer commit y push. Streamlit Cloud redespliega y reindexa automáticamente.

---

## 10. Decisiones de diseño y filosofía

- **Temperatura 0.2:** prioriza precisión y consistencia frente a creatividad. Un reclutador necesita respuestas estables, no variaciones aleatorias del mismo hecho.
- **Respuestas en tercera persona:** el asistente habla *sobre* Alejandro, no se hace pasar *por* él. Esto evita engaños y mantiene la honestidad del proceso.
- **Fallback obligatorio:** si algo no está en la base, el asistente lo admite explícitamente y redirige al contacto directo. Cero alucinaciones permitidas.
- **Memoria por thread:** cada sesión es independiente, lo que evita filtraciones entre reclutadores distintos.

---

## Autor

**Alejandro Rodríguez de la Rosa**
Marketing · Data Science · IA
📧 alejandrorodriguezdelarosa@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/alejandro-rodríguez-de-la-rosa-015956301)
🌐 [portafolio.óptimoia.es](https://portafolio.óptimoia.es)
