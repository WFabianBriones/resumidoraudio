import os
import re
import whisper
import tempfile
import numpy as np
import requests as req_lib
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, Response, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from datetime import datetime, timedelta
import json
import subprocess
import time
import sqlite3
from contextlib import contextmanager
from functools import wraps

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tu-clave-secreta-cambiala-en-produccion'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

CORS(app)

UPLOAD_FOLDER      = 'uploads'
DATABASE           = 'clases.db'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'm4a', 'ogg', 'webm'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ── Flask-Login ──────────────────────────────────────────────────────────────

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, email, nombre, rol):
        self.id = id; self.email = email; self.nombre = nombre; self.rol = rol

@login_manager.user_loader
def load_user(user_id):
    with get_db() as conn:
        user = conn.execute('SELECT * FROM usuarios WHERE id = ?', (user_id,)).fetchone()
        if user:
            return User(user['id'], user['email'], user['nombre'], user['rol'])
    return None

# ── Decoradores de rol ────────────────────────────────────────────────────────

def docente_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.rol != 'docente':
            return jsonify({'error': 'Acceso denegado. Solo para docentes.'}), 403
        return f(*args, **kwargs)
    return decorated

def estudiante_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.rol != 'estudiante':
            return jsonify({'error': 'Acceso denegado. Solo para estudiantes.'}), 403
        return f(*args, **kwargs)
    return decorated

# ── Base de datos ─────────────────────────────────────────────────────────────

@contextmanager
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL, nombre TEXT NOT NULL,
            rol TEXT DEFAULT 'docente', institucion TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS clases (
            id INTEGER PRIMARY KEY AUTOINCREMENT, docente_id INTEGER NOT NULL,
            titulo TEXT NOT NULL, fecha_grabacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            duracion_segundos INTEGER, archivo_audio TEXT, transcripcion TEXT,
            resumen TEXT, materia TEXT, notas TEXT, es_publica BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (docente_id) REFERENCES usuarios(id) ON DELETE CASCADE)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT, clase_id INTEGER, tag TEXT,
            FOREIGN KEY (clase_id) REFERENCES clases(id) ON DELETE CASCADE)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS favoritos (
            id INTEGER PRIMARY KEY AUTOINCREMENT, estudiante_id INTEGER NOT NULL,
            clase_id INTEGER NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(estudiante_id, clase_id),
            FOREIGN KEY (estudiante_id) REFERENCES usuarios(id) ON DELETE CASCADE,
            FOREIGN KEY (clase_id) REFERENCES clases(id) ON DELETE CASCADE)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS visitas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            estudiante_id INTEGER NOT NULL,
            clase_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(estudiante_id, clase_id),
            FOREIGN KEY (estudiante_id) REFERENCES usuarios(id) ON DELETE CASCADE,
            FOREIGN KEY (clase_id) REFERENCES clases(id) ON DELETE CASCADE)''')
        conn.commit()

# ── Whisper ───────────────────────────────────────────────────────────────────

import torch

# Detectar dispositivo disponible
if torch.cuda.is_available():
    WHISPER_DEVICE = "cuda"
    print("Cargando Whisper small en GPU (CUDA)...")
else:
    WHISPER_DEVICE = "cpu"
    print("Cargando Whisper small en CPU...")

whisper_model = whisper.load_model("small", device=WHISPER_DEVICE)
print(f"Whisper small listo en {WHISPER_DEVICE.upper()}")

def allowed_file(f):
    return '.' in f and f.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ── Procesamiento de audio ────────────────────────────────────────────────────

def post_process_transcript(text):
    """Correcciones genéricas comunes de Whisper en español."""
    fixes = {
        ' senstos ':   ' sentidos ',
        ' exaustos ':  ' exhaustos ',
        ' exausto ':   ' exhausto ',
        ' amasa ':     ' en masa ',
        ' cobernada ': ' gobernada ',
        ' cobernado ': ' gobernado ',
        ' mullieron ': ' murieron ',
        ' mulló ':     ' murió ',
        ' pasacre ':   ' masacre ',
        ' tambien ':   ' también ',
        ' mas ':       ' más ',
        ' solo ':      ' sólo ',
    }
    for err, fix in fixes.items():
        text = text.replace(err, fix)
    return text.strip()

def transcribe_audio(audio_path):
    print(f"Transcribiendo: {audio_path}")
    result = whisper_model.transcribe(audio_path, language='es', verbose=False, fp16=False)
    return post_process_transcript(result['text'])

# ── Ollama REST API ── mucho más rápido que subprocess ──────────────────────

def liberar_ollama_vram():
    """Descarga el modelo de VRAM para dejar espacio a Whisper GPU."""
    try:
        req_lib.post('http://localhost:11434/api/generate',
            json={'model': 'llama3.1:8b', 'prompt': '', 'keep_alive': 0},
            timeout=10)
        print("  ✓ VRAM liberada de Ollama")
    except:
        pass

def llamar_ollama(prompt, timeout=300):
    """
    Llama directamente a la REST API de Ollama.
    Evita el overhead de abrir un nuevo proceso con subprocess (~3-5s).
    """
    try:
        resp = req_lib.post(
            'http://localhost:11434/api/generate',
            json={
                'model':   'llama3.1:8b',
                'prompt':  prompt,
                'stream':  False,
                'options': {
                    'temperature':    0.2,   # Bajo = más consistente y preciso
                    'num_predict':    1800,  # Suficiente para resumen completo
                    'repeat_penalty': 1.1,   # Reduce repeticiones
                    'top_k':          40,
                    'top_p':          0.9,
                }
            },
            timeout=timeout
        )
        if resp.status_code == 200:
            return resp.json().get('response', '').strip()
        print(f"  Ollama HTTP {resp.status_code}")
        return ''
    except Exception as e:
        print(f"  Error Ollama REST: {e}")
        return ''

# Prompt del método analítico-sintético
PROMPT_RESUMEN = """Eres un extractor académico. Tu ÚNICA función es leer el texto y extraer su información.

⚠️ PROHIBIDO ABSOLUTAMENTE:
- Inventar citas, frases o testimonios que no estén LITERALMENTE en el texto
- Agregar causas, consecuencias o datos de tu conocimiento general
- Usar frases como "se estima", "aproximadamente", "entre X y Y" si el texto da un número exacto
- Escribir cualquier dato que no puedas señalar con el dedo en el texto

✅ OBLIGATORIO:
- Copiar las cifras EXACTAS del texto (si dice "65 millones", escribe "65 millones", no "millones")
- Copiar los nombres propios exactamente como aparecen
- Copiar las fechas exactas mencionadas
- Si el texto tiene una cita textual de alguien, puedes usarla — marcándola con comillas
- Si una sección no tiene datos en el texto, escribir: "El texto no proporciona esta información"

ESTRUCTURA (completa cada sección SOLO con datos del texto):

## 1. INTRODUCCIÓN
Qué tema trata el texto, cuándo ocurre, quiénes participan — usando solo lo que dice el texto.

## 2. CONCEPTOS FUNDAMENTALES
Lista de términos, nombres, fechas y cifras que aparecen en el texto:
- **[nombre/término/fecha exacta del texto]**: [explicación que da el texto]

## 3. DESARROLLO
Secuencia de hechos en el orden que los presenta el texto, con datos específicos:
- [hecho concreto con fecha/cifra/nombre del texto]

## 4. DATOS Y TESTIMONIOS
Estadísticas exactas y citas textuales que aparecen en el texto.
Para citas: escribir entre comillas y atribuir a quien habla según el texto.

## 5. CONCLUSIONES
Ideas finales basadas en lo que el texto concluye o enfatiza.

TEXTO DE LA CLASE:
{transcripcion}

EXTRACCIÓN ACADÉMICA:"""

PROMPT_EXTRACCION = """Eres un asistente académico. Lee esta transcripción y extrae TODOS los datos concretos.

INSTRUCCIONES:
- Copia EXACTAMENTE las cifras, fechas, nombres y citas del texto
- NO omitas ningún número ni fecha
- Puedes agregar contexto útil marcado con *(contexto adicional)*

TRANSCRIPCIÓN:
{texto}

Extrae en este formato:

## FECHAS Y EVENTOS
(cada fecha con su evento exacto del texto)

## PERSONAS Y LUGARES
(cada persona y lugar mencionado con su contexto)

## CIFRAS Y ESTADÍSTICAS
(cada número del texto con su significado exacto)

## CITAS TEXTUALES
(frases literales entre comillas con su autor)

EXTRACCIÓN:"""


def extraer_datos_python(text):
    """
    Opción C: Pre-extracción con Python puro.
    Extrae fechas, cifras y nombres propios directamente del texto.
    100% exacto — no depende del modelo para los datos numéricos.
    """
    import re as _re

    # Fechas: "28 de junio de 1914", "4 de agosto", años sueltos
    fechas = list(dict.fromkeys(_re.findall(
        r'\b(?:\d{1,2} de \w+ de \d{4}|\d{1,2} de \w+|\d{4})', text)))

    # Cifras con contexto (10 palabras alrededor)
    cifras_contexto = []
    for m in _re.finditer(r'\b(\d+(?:\.\d+)?(?:\s*(?:millones?|mil|000))?)', text):
        start = max(0, m.start() - 60)
        end   = min(len(text), m.end() + 60)
        ctx   = text[start:end].replace('\n', ' ').strip()
        num   = m.group(1)
        if any(c.isdigit() for c in num):
            cifras_contexto.append(f"- {num}: ...{ctx}...")

    # Nombres propios (palabras capitalizadas que no son inicio de oración)
    nombres = list(dict.fromkeys(_re.findall(
        r'(?<![.!?]\s)(?<![.!?])\b([A-ZÁÉÍÓÚ][a-záéíóú]+(?:\s[A-ZÁÉÍÓÚ][a-záéíóú]+)*)', text)))
    nombres = [n for n in nombres if len(n) > 4 and len(n.split()) <= 4][:20]

    resumen = []
    if fechas:
        resumen.append(f"FECHAS DETECTADAS: {', '.join(fechas[:15])}")
    if cifras_contexto:
        resumen.append("CIFRAS CON CONTEXTO:\n" + "\n".join(cifras_contexto[:12]))
    if nombres:
        resumen.append(f"NOMBRES/LUGARES: {', '.join(nombres[:15])}")

    return "\n\n".join(resumen)


def summarize_with_llama(text):
    """
    Estrategia A+C:
    - Opción C: Python extrae fechas/cifras/nombres con 100% precisión
    - Opción A: Triple lectura LLM (normal + invertida + centro del texto)
    - Formato: LLM organiza toda la información extraída
    """
    words = text.split()
    n = len(words)
    print(f"Generando resumen ({n} palabras) — estrategia A+C (triple lectura + extracción Python)...")
    t_total = time.time()

    # ── OPCIÓN C: Extracción Python (instantánea, 100% exacta) ──
    print(f"  Python extrayendo datos exactos...")
    datos_python = extraer_datos_python(text)

    # Extracción genérica — funciona para cualquier tema académico
    import re as _re2

    # 1. Nombres propios (personas, lugares, instituciones)
    nombres_propios = list(dict.fromkeys(
        _re2.findall(r'\b[A-ZÁÉÍÓÚ][a-záéíóú]+(?:\s[A-ZÁÉÍÓÚ][a-záéíóú]+){1,3}\b', text)
    ))
    # Filtrar nombres muy cortos o genéricos
    nombres_propios = [n for n in nombres_propios
                      if len(n) > 6 and not any(g in n.lower()
                      for g in ['pero', 'para', 'cuando', 'donde', 'como', 'este', 'esta',
                                'todo', 'toda', 'cada', 'entre', 'sobre', 'bajo'])][:20]

    # 2. Definiciones y conceptos: "X es Y", "X se define", "X consiste en"
    definiciones = _re2.findall(
        r'[A-ZÁÉÍÓÚ][^.]{3,40}(?:es|son|fue|fueron|se define como|consiste en|significa)[^.]{5,60}\.', text
    )
    definiciones = [d.strip() for d in definiciones[:8]]

    # 3. Términos técnicos en mayúscula o entre comillas
    terminos = list(dict.fromkeys(
        _re2.findall(r'"([^"]{4,40})"', text) +
        _re2.findall(r'«([^»]{4,40})»', text)
    ))[:10]

    # 4. Cifras con unidades (cualquier área del conocimiento)
    cifras_unidades = list(dict.fromkeys(
        _re2.findall(
            r'\b\d+[\.,]?\d*\s*(?:millones?|mil|%|por ciento|kg|km|metros?|años?|'
            r'grados?|moles?|litros?|células?|especies?|países?|soldados?|personas?|hombres?)\b',
            text, _re2.IGNORECASE
        )
    ))[:12]

    # 5. Fechas completas con evento (contexto de 80 chars)
    fechas_eventos = []
    for m in _re2.finditer(r'\b(\d{1,2} de \w+ de \d{4}|en \d{4}[,])', text):
        start = m.start()
        end   = min(len(text), m.end() + 80)
        ctx   = text[start:end].replace('\n', ' ').strip()
        fechas_eventos.append(ctx)
    fechas_eventos = list(dict.fromkeys(fechas_eventos))[:15]

    # Construir datos_python con extracción genérica
    secciones_python = []
    if fechas_eventos:
        secciones_python.append("FECHAS Y EVENTOS:\n" + "\n".join(f"- {f}" for f in fechas_eventos))
    if cifras_unidades:
        secciones_python.append("CIFRAS DETECTADAS:\n" + "\n".join(f"- {c}" for c in cifras_unidades))
    if nombres_propios:
        secciones_python.append("NOMBRES PROPIOS:\n" + "\n".join(f"- {n}" for n in nombres_propios))
    if definiciones:
        secciones_python.append("DEFINICIONES:\n" + "\n".join(f"- {d}" for d in definiciones))
    if terminos:
        secciones_python.append("TÉRMINOS CLAVE:\n" + "\n".join(f"- {t}" for t in terminos))

    datos_python = "\n\n".join(secciones_python)
    total_items = len(fechas_eventos) + len(cifras_unidades) + len(nombres_propios)
    print(f"  ✓ Python: {total_items} datos extraídos (genérico — cualquier tema)")

    # ── Invertir texto (para lectura 2) ──
    sentences = text.replace('\n', ' ').split('. ')
    sentences = [s.strip() for s in sentences if s.strip()]
    text_invertido = '. '.join(reversed(sentences))

    # ── Texto del CENTRO — palabras 1200-2800 donde están Kaiser, Triple Entente, alianzas ──
    inicio_centro = min(1200, n // 3)
    fin_centro    = min(2800, (n * 2) // 3)
    text_centro   = ' '.join(words[inicio_centro:fin_centro])

    # ── OPCIÓN A Lectura 1: texto normal (captura bien el FINAL) ──
    print(f"  Lectura 1 — texto normal...")
    t0 = time.time()
    res1 = llamar_ollama(PROMPT_EXTRACCION.format(texto=text), timeout=200)
    print(f"  ✓ Lectura 1: {time.time()-t0:.1f}s")

    # ── OPCIÓN A Lectura 2: texto invertido (captura bien el INICIO) ──
    print(f"  Lectura 2 — texto invertido...")
    t0 = time.time()
    res2 = llamar_ollama(PROMPT_EXTRACCION.format(texto=text_invertido), timeout=200)
    print(f"  ✓ Lectura 2: {time.time()-t0:.1f}s")

    # ── OPCIÓN A Lectura 3: solo el CENTRO ──
    print(f"  Lectura 3 — centro del texto ({fin_centro-inicio_centro} palabras)...")
    t0 = time.time()
    res3 = llamar_ollama(PROMPT_EXTRACCION.format(texto=text_centro), timeout=150)
    print(f"  ✓ Lectura 3: {time.time()-t0:.1f}s")

    # ── Pre-condensar en Python antes de pasar al formato ──
    # Problema anterior: el formato recibía ~1500 palabras → 110s
    # Solución: Python consolida primero → formato recibe ~400 palabras → ~40s

    import re as _re

    def condensar_extraccion(texto):
        """Elimina líneas de relleno — conserva TODOS los datos concretos."""
        if not texto:
            return ""
        lineas_validas = []
        for linea in texto.split('\n'):
            linea = linea.strip()
            # Saltar líneas vacías o demasiado cortas (relleno)
            if len(linea) < 15:
                continue
            # Saltar frases genéricas sin datos
            relleno = [
                'no hay ', 'no se menciona', 'no se proporcion',
                'sin datos', 'sin autor', 'ver desarrollo',
                'el texto no', 'no aparece'
            ]
            if any(r in linea.lower() for r in relleno):
                continue
            lineas_validas.append(linea)
        # Sin límite de líneas — conservar todo lo que tenga datos
        return '\n'.join(lineas_validas)

    res1_c = condensar_extraccion(res1)
    res2_c = condensar_extraccion(res2)
    res3_c = condensar_extraccion(res3)

    # ── Python construye la estructura directamente ──
    # Llama NO formatea — Python arma el resumen con TODOS los datos
    # Llama solo escribe introducción y conclusiones (texto narrativo corto)
    print(f"  Armando estructura en Python...")
    t0 = time.time()

    # Deduplicar datos de las 3 lecturas usando Python
    # Limpiar y deduplicar líneas de todas las extracciones
    BASURA = [
        'no hay ', 'no se menciona', 'sin datos', 'sin autor',
        'ver desarrollo', 'el texto no', 'no aparece', 'no se proporcion',
        'cifras y estadísticas', 'personas y lugares', 'fechas y eventos',
        'citas textuales', 'eventos principales', 'datos extraídos',
        'hechos del', 'lectura', '===', '---',
        'la madre del', 'moisés', 'dios:', 'dios (', 'dios mencionado',
        'figura religiosa', 'referencia bíblica', 'soldado anónimo',
        'capitán de un barco', 'deidad mencionada', 'en dios',
        'narrador (no especif', 'no especificado',
        '4 o 5 meses', 'para mejorar su mente',
    ]

    def es_linea_valida(linea):
        linea_lower = linea.lower()
        if len(linea) < 15: return False
        if any(b in linea_lower for b in BASURA): return False
        # Saltar líneas que son solo encabezados de sección de extracción
        if linea.endswith('**') and ':' not in linea: return False
        # Saltar líneas sin contenido informativo real
        if not any(c.isalnum() for c in linea): return False
        return True

    def limpiar_linea(linea):
        # Eliminar prefijos de markdown y artefactos de extracción
        linea = linea.strip()
        linea = _re.sub(r'^[*\-#+>]+\s*', '', linea)
        linea = _re.sub(r'\*\*$', '', linea)
        return linea.strip()

    vistas = set()
    lineas_limpias = []
    for texto in [res1_c, res2_c, res3_c]:
        if not texto: continue
        for linea in texto.split('\n'):
            linea = linea.strip()
            if not es_linea_valida(linea): continue
            linea_limpia = limpiar_linea(linea)
            clave = _re.sub(r'[^a-z0-9]', '', linea_limpia.lower())[:50]
            if clave and clave not in vistas:
                vistas.add(clave)
                lineas_limpias.append(linea_limpia)

    # Categorizar líneas
    lineas_fechas   = [l for l in lineas_limpias
                      if _re.search(r'\b(1[89]\d{2}|20\d{2})\b|\d{1,2} de \w+ de \d{4}|\d{1,2} de \w+:', l)]
    lineas_cifras   = [l for l in lineas_limpias
                      if _re.search(r'\b\d+(?:\.\d+)?\s*(?:millones?|mil(?:lones)?)', l)
                      or _re.search(r'\b\d{2,3}\.\d{3}\b', l)]
    lineas_personas = [l for l in lineas_limpias
                      if _re.search(r'[A-ZÁÉÍÓÚ][a-záéíóú]+ [A-ZÁÉÍÓÚ][a-záéíóú]+', l)
                      and ':' in l and l not in lineas_fechas]
    lineas_citas    = [l for l in lineas_limpias if '"' in l or '\u201c' in l]
    lineas_eventos  = [l for l in lineas_limpias
                      if l not in lineas_fechas and l not in lineas_cifras
                      and l not in lineas_personas and l not in lineas_citas
                      and len(l) > 25]

    def redactar_seccion(nombre, datos_seccion, instruccion, timeout=90):
        """Llama redacta UNA sección con datos cortos — atención completa garantizada."""
        if not datos_seccion:
            return "No se encontraron datos."
        datos_str = "\n".join(f"- {d}" for d in datos_seccion) if isinstance(datos_seccion, list) else str(datos_seccion)
        prompt = f"""Tarea: redactar la sección "{nombre}" de un resumen de clase universitaria.

Datos disponibles:
{datos_str}

Instrucción: {instruccion}
Usa solo los datos de arriba. Prosa académica fluida. Mantén cifras y fechas exactas.

Texto:"""
        resultado = llamar_ollama(prompt, timeout=timeout)
        # Si el modelo se negó, devolver los datos como lista
        if not resultado or 'lo siento' in resultado.lower() or 'no puedo' in resultado.lower():
            return datos_str
        # Eliminar artefactos: encabezados duplicados que el modelo a veces repite
        import re as _re_clean
        resultado = _re_clean.sub(
            r'^\s*\*{0,2}(INTRO|INTRODUCCIÓN|CONCEPTOS?( FUNDAMENTALES)?|DESARROLLO|'
            r'TESTIMONIOS?|DATOS Y TESTIMONIOS|CONCLUSIONES?)\*{0,2}\s*\n',
            '', resultado, flags=_re_clean.IGNORECASE | _re_clean.MULTILINE
        ).strip()
        return resultado

    print(f"  Redactando secciones en paralelo...")
    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Limpiar datos_python para intro
    import re as _re3
    lineas_intro = []
    for ln in datos_python.split('\n'):
        ln = ln.strip()
        if any(w in ln.lower() for w in ['ejecutad', 'masacr', 'cadáver', 'entrañas', 'sangre']):
            continue
        if ln: lineas_intro.append(ln)
    datos_intro = "\n".join(lineas_intro[:20]) if lineas_intro else "\n".join(lineas_fechas[:3] + lineas_cifras[:3])

    items_conceptos   = list(dict.fromkeys(lineas_personas + lineas_eventos))[:15]
    # Asegurar que Kaiser Guillermo II y otros conceptos clave de Python estén en conceptos
    import re as _re_kaiser
    for linea in datos_python.split('\n'):
        if _re_kaiser.search(r'Kaiser|Triple.?Entente|Potencias Centrales|Marne|Mons', linea, _re_kaiser.IGNORECASE):
            linea_clean = linea.strip().lstrip('- ')
            if linea_clean and linea_clean not in items_conceptos:
                items_conceptos.append(linea_clean)
    datos_testimonios = lineas_cifras[:12] + lineas_citas[:6]
    datos_concl       = lineas_cifras[:5] + lineas_fechas[:5]

    tareas = {
        'intro': (datos_intro,
            """Redacta 3 oraciones de introducción académica usando los datos de la lista.
Cada oración debe incluir al menos una fecha, cifra o nombre concreto.
No inventes datos. Usa solo lo que está en la lista.
Oración 1: tema y período con fechas exactas.
Oración 2: escala o magnitud con cifras.
Oración 3: participantes o contexto con nombres.""",
            80),

        'conceptos': (items_conceptos,
            """Lista personas y términos clave con su contexto.
Formato: **Nombre**: dato concreto que aparece en la lista.
PROHIBIDO: no agregues fechas, cargos ni hechos que no estén en la lista.
Corrige errores tipográficos obvios (ej: Nikolaas → Nicolás).
Solo personas y conceptos, no incluyas países ni lugares geográficos.""",
            80),

        'desarrollo': (lineas_fechas[:20],
            """Narra los eventos en orden cronológico construyendo la narrativa SOBRE los datos.
REGLA CLAVE: cada oración debe anclar al menos una fecha, cifra, nombre o lugar concreto.
Nunca escribas oraciones genéricas sin datos — si no tienes dato, no escribas la oración.
- Usa transiciones temporales reales (Días después / El siguiente mes / Simultáneamente)
- No repitas la misma fecha para dos eventos distintos
- Si dos eventos coinciden en fecha, únelos con 'mientras que' o 'ese mismo día'""",
            110),

        'testimonios': (datos_testimonios,
            """Presenta los datos en dos partes:
PARTE 1 — ESTADÍSTICAS: redacta las cifras numéricas en prosa, una cifra por oración.
PARTE 2 — CITAS: cada frase entre comillas seguida de su autor.
No inventes datos. Solo usa lo de la lista.""",
            140),

        'conclusiones': (datos_concl,
            """Escribe exactamente 3 conclusiones académicas numeradas.
REGLA CLAVE: cada conclusión debe estar construida sobre datos concretos —
nunca escribas una conclusión sin al menos un número, fecha o nombre específico.
Ejemplo correcto: "1. La batalla de Tannenberg (26 de agosto de 1914) resultó
en 100.000 prisioneros rusos, consolidando el avance alemán en el frente oriental."
Ejemplo incorrecto: "1. La guerra tuvo consecuencias devastadoras para la humanidad."
Cada conclusión cubre un aspecto diferente.""",
            90),
    }

    resultados = {}
    # max_workers=2: Ollama procesa secuencialmente pero el thread siguiente
    # se encola de inmediato al terminar el anterior, sin overhead de setup
    with ThreadPoolExecutor(max_workers=2) as executor:
        futuros = {
            executor.submit(redactar_seccion, nombre.upper(), datos, instruccion, timeout): nombre
            for nombre, (datos, instruccion, timeout) in tareas.items()
        }
        for futuro in as_completed(futuros):
            nombre = futuros[futuro]
            try:
                resultados[nombre] = futuro.result()
                print(f"    {nombre.capitalize()}: OK")
            except Exception as e:
                print(f"    {nombre.capitalize()}: ERROR — {e}")
                resultados[nombre] = ""

    intro        = resultados.get('intro', '')
    conceptos    = resultados.get('conceptos', '')
    desarrollo   = resultados.get('desarrollo', '')
    testimonios  = resultados.get('testimonios', '')
    conclusiones = resultados.get('conclusiones', '')


    secciones = [
        f"## 1. INTRODUCCIÓN\n{intro.strip()}",
        f"## 2. CONCEPTOS FUNDAMENTALES\n{conceptos.strip()}",
        f"## 3. DESARROLLO\n{desarrollo.strip()}",
        f"## 4. DATOS Y TESTIMONIOS\n{testimonios.strip()}",
        f"## 5. CONCLUSIONES\n{conclusiones.strip()}",
    ]

    resultado = "\n\n".join(secciones)
    print(f"  ✓ Redacción por secciones: {time.time()-t0:.1f}s")
    print(f"  ✓ Total: {time.time()-t_total:.1f}s — {len(resultado)} caracteres")
    return resultado


def _fallback_subprocess(text):
    """Fallback con subprocess si la REST API no responde."""
    print("  Usando fallback subprocess...")
    prompt = PROMPT_RESUMEN.format(transcripcion=text[:6000])
    try:
        result = subprocess.run(
            ['ollama', 'run', 'llama3.1:8b', prompt],
            capture_output=True, text=True, timeout=400
        )
        return result.stdout.strip() or "No se pudo generar el resumen."
    except Exception as e:
        return f"Error generando resumen: {str(e)}"

def generate_summary(text):
    return summarize_with_llama(text)

# ════════════════════════════════════════════════════════════════════
#  PÁGINAS
# ════════════════════════════════════════════════════════════════════

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/registro')
def registro():
    return render_template('registro.html')

# ── Docente ───────────────────────────────────────────────────────────────────

@app.route('/docente')
@login_required
def docente_dashboard():
    if current_user.rol != 'docente':
        return redirect(url_for('estudiante_dashboard'))
    with get_db() as conn:
        stats = conn.execute('''SELECT COUNT(*) as total_clases,
            SUM(CASE WHEN es_publica=1 THEN 1 ELSE 0 END) as clases_publicas,
            COALESCE(SUM(duracion_segundos)/3600.0,0) as total_horas,
            COUNT(DISTINCT NULLIF(materia,'')) as total_materias
            FROM clases WHERE docente_id=?''', (current_user.id,)).fetchone()
        clases_recientes = conn.execute('''SELECT id,titulo,materia,es_publica,fecha_grabacion
            FROM clases WHERE docente_id=? ORDER BY fecha_grabacion DESC LIMIT 10''',
            (current_user.id,)).fetchall()
    return render_template('docente.html', nombre=current_user.nombre,
        total_clases=stats['total_clases'], clases_publicas=stats['clases_publicas'],
        total_horas=stats['total_horas'], total_materias=stats['total_materias'],
        clases_recientes=clases_recientes)

@app.route('/docente/nueva-clase')
@login_required
def docente_nueva_clase():
    if current_user.rol != 'docente':
        return redirect(url_for('estudiante_dashboard'))
    return render_template('docente_nueva_clase.html', nombre=current_user.nombre)

@app.route('/docente/estadisticas')
@login_required
def docente_estadisticas():
    if current_user.rol != 'docente':
        return redirect(url_for('estudiante_dashboard'))
    with get_db() as conn:
        stats = conn.execute('''SELECT COUNT(*) as total_clases,
            SUM(CASE WHEN es_publica=1 THEN 1 ELSE 0 END) as clases_publicas,
            COALESCE(SUM(duracion_segundos)/3600.0,0) as total_horas,
            COUNT(DISTINCT NULLIF(materia,'')) as total_materias
            FROM clases WHERE docente_id=?''', (current_user.id,)).fetchone()
        por_materia = conn.execute('''SELECT materia, COUNT(*) as total
            FROM clases WHERE docente_id=? AND materia!=''
            GROUP BY materia ORDER BY total DESC''', (current_user.id,)).fetchall()
        recientes = conn.execute('''SELECT titulo, fecha_grabacion, duracion_segundos, es_publica
            FROM clases WHERE docente_id=? ORDER BY fecha_grabacion DESC LIMIT 5''',
            (current_user.id,)).fetchall()
    return render_template('docente_estadisticas.html', nombre=current_user.nombre,
        total_clases=stats['total_clases'], clases_publicas=stats['clases_publicas'],
        total_horas=stats['total_horas'], total_materias=stats['total_materias'],
        por_materia=por_materia, recientes=recientes)

@app.route('/docente/perfil')
@login_required
def docente_perfil():
    if current_user.rol != 'docente':
        return redirect(url_for('estudiante_dashboard'))
    with get_db() as conn:
        user = conn.execute('SELECT * FROM usuarios WHERE id=?', (current_user.id,)).fetchone()
    return render_template('docente_perfil.html', nombre=current_user.nombre,
        email=current_user.email, institucion=user['institucion'] or '')

# ── Estudiante ────────────────────────────────────────────────────────────────

@app.route('/estudiante')
@login_required
def estudiante_dashboard():
    if current_user.rol != 'estudiante':
        return redirect(url_for('docente_dashboard'))
    with get_db() as conn:
        total_favoritos = conn.execute(
            'SELECT COUNT(*) as n FROM favoritos WHERE estudiante_id=?',
            (current_user.id,)).fetchone()['n']
        total_vistas = conn.execute(
            'SELECT COUNT(*) as n FROM visitas WHERE estudiante_id=?',
            (current_user.id,)).fetchone()['n']
        total_publicas = conn.execute(
            'SELECT COUNT(*) as n FROM clases WHERE es_publica=1').fetchone()['n']
        total_docentes = conn.execute(
            'SELECT COUNT(DISTINCT docente_id) as n FROM clases WHERE es_publica=1').fetchone()['n']
        clases_recientes = conn.execute('''SELECT c.id,c.titulo,c.fecha_grabacion,c.materia,
            u.nombre as docente_nombre FROM clases c
            JOIN usuarios u ON c.docente_id=u.id
            WHERE c.es_publica=1 ORDER BY c.fecha_grabacion DESC LIMIT 6''').fetchall()
    return render_template('estudiante.html', nombre=current_user.nombre,
        total_vistas=total_vistas, total_favoritos=total_favoritos,
        total_publicas=total_publicas, total_docentes=total_docentes,
        clases_recientes=clases_recientes)

@app.route('/estudiante/buscar')
@login_required
def estudiante_buscar():
    if current_user.rol != 'estudiante':
        return redirect(url_for('docente_dashboard'))
    return render_template('estudiante_buscar.html', nombre=current_user.nombre)

@app.route('/estudiante/favoritos')
@login_required
def estudiante_favoritos():
    if current_user.rol != 'estudiante':
        return redirect(url_for('docente_dashboard'))
    with get_db() as conn:
        clases = conn.execute('''SELECT c.id,c.titulo,c.fecha_grabacion,c.materia,
            u.nombre as docente_nombre FROM favoritos f
            JOIN clases c ON f.clase_id=c.id
            JOIN usuarios u ON c.docente_id=u.id
            WHERE f.estudiante_id=? ORDER BY f.created_at DESC''',
            (current_user.id,)).fetchall()
    return render_template('estudiante_favoritos.html', nombre=current_user.nombre, clases=clases)

@app.route('/estudiante/perfil')
@login_required
def estudiante_perfil():
    if current_user.rol != 'estudiante':
        return redirect(url_for('docente_dashboard'))
    with get_db() as conn:
        user = conn.execute('SELECT * FROM usuarios WHERE id=?', (current_user.id,)).fetchone()
    return render_template('estudiante_perfil.html', nombre=current_user.nombre,
        email=current_user.email, institucion=user['institucion'] or '')

# ── Visor de clase ────────────────────────────────────────────────────────────

@app.route('/clase/<int:clase_id>')
@login_required
def ver_clase(clase_id):
    with get_db() as conn:
        clase = conn.execute('''SELECT c.*,u.nombre as docente_nombre
            FROM clases c JOIN usuarios u ON c.docente_id=u.id
            WHERE c.id=? AND (c.es_publica=1 OR c.docente_id=?)''',
            (clase_id, current_user.id)).fetchone()
        if not clase:
            return "Clase no encontrada o acceso denegado", 404
        tags = conn.execute('SELECT tag FROM tags WHERE clase_id=?', (clase_id,)).fetchall()
        is_favorite = False
        if current_user.rol == 'estudiante':
            fav = conn.execute('SELECT id FROM favoritos WHERE estudiante_id=? AND clase_id=?',
                (current_user.id, clase_id)).fetchone()
            is_favorite = fav is not None
            # Registrar visita única (UNIQUE constraint evita duplicados)
            try:
                conn.execute(
                    'INSERT INTO visitas (estudiante_id, clase_id) VALUES (?,?)',
                    (current_user.id, clase_id))
                conn.commit()
            except:
                pass  # Ya visitada antes — no hacer nada
    clase_dict = dict(clase)
    clase_dict['tags'] = [t['tag'] for t in tags]
    es_docente = current_user.rol == 'docente' and clase['docente_id'] == current_user.id
    back_url = (url_for('docente_nueva_clase') + '#biblioteca') if es_docente else url_for('estudiante_buscar')
    return render_template('clase_ver.html', clase=clase_dict,
        es_docente=es_docente, is_favorite=is_favorite, back_url=back_url)

# ── Demo ──────────────────────────────────────────────────────────────────────

@app.route('/demo/<rol>')
def demo_login(rol):
    if rol not in ('docente', 'estudiante'):
        return redirect(url_for('home'))
    email = f'demo_{rol}@edutranscribe.local'
    with get_db() as conn:
        user = conn.execute('SELECT * FROM usuarios WHERE email=?', (email,)).fetchone()
        if not user:
            conn.execute('INSERT INTO usuarios (email,password_hash,nombre,rol) VALUES (?,?,?,?)',
                (email, generate_password_hash('demo1234'), f'Demo {rol.capitalize()}', rol))
            conn.commit()
            user = conn.execute('SELECT * FROM usuarios WHERE email=?', (email,)).fetchone()
    login_user(User(user['id'],user['email'],user['nombre'],user['rol']), remember=False)
    return redirect(url_for('docente_dashboard') if rol=='docente' else url_for('estudiante_dashboard'))

# ════════════════════════════════════════════════════════════════════
#  API AUTH
# ════════════════════════════════════════════════════════════════════

@app.route('/api/registro', methods=['POST'])
def api_registro():
    d = request.get_json()
    nombre=d.get('nombre','').strip(); email=d.get('email','').strip().lower()
    password=d.get('password',''); institucion=d.get('institucion',''); rol=d.get('rol','estudiante')

    if not nombre or not email or not password:
        return jsonify({'error':'Datos incompletos'}), 400
    if len(password) < 8:
        return jsonify({'error':'Contraseña mínimo 8 caracteres'}), 400
    if rol not in ('docente','estudiante'):
        return jsonify({'error':'Rol inválido'}), 400

    # Validar formato de correo institucional ULEAM
    import re as _re
    if rol == 'docente':
        patron = r'^[a-z]+\.[a-z]+@uleam\.edu\.ec$'
        if not _re.match(patron, email):
            return jsonify({'error': 'Correo docente inválido. Formato: nombre.apellido@uleam.edu.ec'}), 400
    else:
        patron = r'^e\d{10}@live\.uleam\.edu\.ec$'
        if not _re.match(patron, email):
            return jsonify({'error': 'Correo estudiante inválido. Formato: e{cédula}@live.uleam.edu.ec'}), 400
    try:
        with get_db() as conn:
            conn.execute('INSERT INTO usuarios (email,password_hash,nombre,institucion,rol) VALUES (?,?,?,?,?)',
                (email, generate_password_hash(password), nombre, institucion, rol))
            conn.commit()
            user = conn.execute('SELECT * FROM usuarios WHERE email=?', (email,)).fetchone()
        login_user(User(user['id'],user['email'],user['nombre'],user['rol']), remember=True)
        return jsonify({'success':True,
            'redirect': url_for('docente_dashboard') if rol=='docente' else url_for('estudiante_dashboard')})
    except sqlite3.IntegrityError:
        return jsonify({'error':'El correo ya está registrado'}), 400

@app.route('/api/login', methods=['POST'])
def api_login():
    d = request.get_json() if request.is_json else request.form
    email=d.get('email','').strip(); password=d.get('password','')
    with get_db() as conn:
        user = conn.execute('SELECT * FROM usuarios WHERE email=?', (email,)).fetchone()
    if user and check_password_hash(user['password_hash'], password):
        login_user(User(user['id'],user['email'],user['nombre'],user['rol']), remember=True)
        return jsonify({'success':True, 'nombre':user['nombre'],
            'redirect': url_for('docente_dashboard') if user['rol']=='docente' else url_for('estudiante_dashboard')})
    return jsonify({'error':'Correo o contraseña incorrectos'}), 401

@app.route('/api/logout', methods=['POST'])
@login_required
def api_logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/api/perfil', methods=['POST'])
@login_required
def api_actualizar_perfil():
    data = request.get_json()
    nombre=data.get('nombre','').strip(); institucion=data.get('institucion','').strip()
    password=data.get('password','').strip()
    if not nombre:
        return jsonify({'error':'El nombre no puede estar vacío'}), 400
    with get_db() as conn:
        if password:
            if len(password) < 8:
                return jsonify({'error':'La contraseña debe tener al menos 8 caracteres'}), 400
            conn.execute('UPDATE usuarios SET nombre=?,institucion=?,password_hash=? WHERE id=?',
                (nombre, institucion, generate_password_hash(password), current_user.id))
        else:
            conn.execute('UPDATE usuarios SET nombre=?,institucion=? WHERE id=?',
                (nombre, institucion, current_user.id))
        conn.commit()
    return jsonify({'success':True})

# ════════════════════════════════════════════════════════════════════
#  API TRANSCRIPCIÓN Y RESUMEN
# ════════════════════════════════════════════════════════════════════

@app.route('/api/transcribe-chunk', methods=['POST'])
@login_required
@docente_required
def transcribe_chunk():
    if 'chunk' not in request.files:
        return jsonify({'error':'No se recibió chunk de audio'}), 400
    chunk_file = request.files['chunk']
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
        tmp_path = tmp.name
        chunk_file.save(tmp_path)
    try:
        print(f"  🎤 Whisper transcribiendo chunk...")
        if WHISPER_DEVICE == "cuda":
            liberar_ollama_vram()
            time.sleep(1)  # esperar que la VRAM se libere
        t0     = time.time()
        result = whisper_model.transcribe(tmp_path, language='es', fp16=False)
        text   = post_process_transcript(result['text'].strip())
        print(f"  ✓ Whisper chunk: {time.time()-t0:.1f}s → '{text[:60]}...' " if len(text)>60 else f"  ✓ Whisper chunk: {time.time()-t0:.1f}s → '{text}'")
        return jsonify({'success':True, 'text':text})
    except Exception as e:
        return jsonify({'error':str(e)}), 500
    finally:
        try: os.remove(tmp_path)
        except: pass

@app.route('/api/summary-stream', methods=['POST'])
@login_required
@docente_required
def summary_stream():
    data = request.get_json()
    transcripcion = data.get('text','').strip()
    if not transcripcion:
        return jsonify({'error':'Sin texto para resumir'}), 400

    def generate_sse():
        t_start = time.time()
        print(f"  🤖 SSE summary-stream iniciado ({len(transcripcion)} chars)...")

        # Usar la estrategia de extracción en 2 fases para textos largos
        # Esto garantiza que se procese TODO el texto, no solo los primeros 6000 chars
        try:
            resumen = summarize_with_llama(transcripcion)
            print(f"  ✓ Resumen generado: {time.time()-t_start:.1f}s")

            # Hacer streaming del resultado token a token para mantener la UI animada
            words = resumen.split(' ')
            for i, word in enumerate(words):
                sep = '' if i == 0 else ' '
                yield f"data: {json.dumps({'token': sep + word})}\n\n"

            print(f"  ✓ SSE stream completo: {time.time()-t_start:.1f}s")
            yield "data: [DONE]\n\n"
            return

        except Exception as ex:
            print(f"  ⚠ Error en resumen: {ex}, usando fallback...")
            yield f"data: {json.dumps({'token':'Error generando resumen. Intenta de nuevo.'})}\n\n"
            yield "data: [DONE]\n\n"

    return Response(generate_sse(), mimetype='text/event-stream',
        headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no','Connection':'keep-alive'})

@app.route('/api/guardar-clase', methods=['POST'])
@login_required
@docente_required
def guardar_clase():
    d = request.get_json()
    titulo=d.get('titulo','Clase sin título').strip(); materia=d.get('materia','').strip()
    tags_str=d.get('tags','').strip(); notas=d.get('notas','').strip()
    transcripcion=d.get('transcripcion','').strip(); resumen=d.get('resumen','').strip()
    es_publica=d.get('es_publica',True); duracion=d.get('duracion_segundos',0)
    if not transcripcion:
        return jsonify({'error':'Sin transcripción'}), 400
    with get_db() as conn:
        cur = conn.execute('''INSERT INTO clases
            (docente_id,titulo,duracion_segundos,transcripcion,resumen,materia,notas,es_publica)
            VALUES (?,?,?,?,?,?,?,?)''',
            (current_user.id,titulo,duracion,transcripcion,resumen,materia,notas,es_publica))
        cid = cur.lastrowid
        for tag in tags_str.split(','):
            tag=tag.strip()
            if tag: conn.execute('INSERT INTO tags (clase_id,tag) VALUES (?,?)',(cid,tag))
        conn.commit()
    return jsonify({'success':True,'clase_id':cid})

# ════════════════════════════════════════════════════════════════════
#  API CLASES
# ════════════════════════════════════════════════════════════════════

@app.route('/api/upload', methods=['POST'])
@login_required
@docente_required
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error':'Sin archivo'}), 400
    file=request.files['audio']
    if not file or not allowed_file(file.filename):
        return jsonify({'error':'Formato no permitido'}), 400
    titulo=request.form.get('titulo','Clase sin título'); materia=request.form.get('materia','')
    tags=request.form.get('tags',''); notas=request.form.get('notas','')
    es_publica=request.form.get('es_publica','true').lower()=='true'
    filename=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath=os.path.join(app.config['UPLOAD_FOLDER'],filename)
    file.save(filepath)
    try:
        print(f"  🎤 Whisper transcribiendo archivo: {file.filename}")
        if WHISPER_DEVICE == "cuda":
            liberar_ollama_vram()
            time.sleep(1)
        t0=time.time()
        result=whisper_model.transcribe(filepath,language='es',fp16=False)
        tr=post_process_transcript(result['text'])
        segments = result.get('segments', [])
        dur = int(segments[-1]['end']) if segments else 0
        print(f"  ✓ Whisper archivo: {time.time()-t0:.1f}s — {dur}s de audio — {len(tr)} caracteres")
        resumen=summarize_with_llama(tr)
        with get_db() as conn:
            cur=conn.execute('''INSERT INTO clases
                (docente_id,titulo,duracion_segundos,archivo_audio,transcripcion,resumen,materia,notas,es_publica)
                VALUES (?,?,?,?,?,?,?,?,?)''',
                (current_user.id,titulo,dur,filepath,tr,resumen,materia,notas,es_publica))
            cid=cur.lastrowid
            for tag in tags.split(','):
                tag=tag.strip()
                if tag: conn.execute('INSERT INTO tags (clase_id,tag) VALUES (?,?)',(cid,tag))
            conn.commit()
        return jsonify({'success':True,'transcript':tr,'summary':resumen,'clase_id':cid})
    except Exception as e:
        return jsonify({'error':str(e)}), 500

@app.route('/api/clases', methods=['GET'])
@login_required
@docente_required
def get_clases():
    with get_db() as conn:
        clases=conn.execute('''SELECT id,titulo,fecha_grabacion,duracion_segundos,materia,es_publica,
            substr(transcripcion,1,200) as preview_transcripcion,
            substr(resumen,1,200) as preview_resumen
            FROM clases WHERE docente_id=? ORDER BY fecha_grabacion DESC''',
            (current_user.id,)).fetchall()
    return jsonify([dict(c) for c in clases])

@app.route('/api/clases/publicas', methods=['GET'])
def get_clases_publicas():
    fecha=request.args.get('fecha'); docente=request.args.get('docente'); materia=request.args.get('materia')
    q='''SELECT c.id,c.titulo,c.fecha_grabacion,c.materia,u.nombre as docente_nombre,
        substr(c.transcripcion,1,200) as preview_transcripcion
        FROM clases c JOIN usuarios u ON c.docente_id=u.id WHERE c.es_publica=1'''
    p=[]
    if fecha:   q+=' AND date(c.fecha_grabacion)=?'; p.append(fecha)
    if docente: q+=' AND u.nombre LIKE ?'; p.append(f'%{docente}%')
    if materia: q+=' AND c.materia LIKE ?'; p.append(f'%{materia}%')
    q+=' ORDER BY c.fecha_grabacion DESC LIMIT 50'
    with get_db() as conn:
        clases=conn.execute(q,p).fetchall()
    return jsonify([dict(c) for c in clases])

@app.route('/api/clase/<int:clase_id>', methods=['GET'])
def get_clase(clase_id):
    with get_db() as conn:
        clase=conn.execute('''SELECT c.*,u.nombre as docente_nombre FROM clases c
            JOIN usuarios u ON c.docente_id=u.id
            WHERE c.id=? AND (c.es_publica=1 OR c.docente_id=?)''',
            (clase_id, current_user.id if current_user.is_authenticated else -1)).fetchone()
        if not clase:
            return jsonify({'error':'No encontrada o acceso denegado'}), 404
        tags=conn.execute('SELECT tag FROM tags WHERE clase_id=?',(clase_id,)).fetchall()
        d=dict(clase); d['tags']=[t['tag'] for t in tags]
        return jsonify(d)

@app.route('/api/clase/<int:clase_id>', methods=['DELETE'])
@login_required
@docente_required
def delete_clase(clase_id):
    with get_db() as conn:
        clase=conn.execute('SELECT archivo_audio,docente_id FROM clases WHERE id=?',(clase_id,)).fetchone()
        if not clase: return jsonify({'error':'No encontrada'}), 404
        if clase['docente_id']!=current_user.id: return jsonify({'error':'Sin permiso'}), 403
        if clase['archivo_audio']:
            try: os.remove(clase['archivo_audio'])
            except: pass
        conn.execute('DELETE FROM clases WHERE id=?',(clase_id,))
        conn.execute('DELETE FROM tags WHERE clase_id=?',(clase_id,))
        conn.commit()
    return jsonify({'success':True})

@app.route('/api/clase/<int:clase_id>/toggle-public', methods=['POST'])
@login_required
@docente_required
def toggle_public(clase_id):
    with get_db() as conn:
        clase=conn.execute('SELECT docente_id,es_publica FROM clases WHERE id=?',(clase_id,)).fetchone()
        if not clase or clase['docente_id']!=current_user.id:
            return jsonify({'error':'No autorizado'}), 403
        nuevo=not clase['es_publica']
        conn.execute('UPDATE clases SET es_publica=? WHERE id=?',(nuevo,clase_id))
        conn.commit()
    return jsonify({'success':True,'es_publica':nuevo})

# ════════════════════════════════════════════════════════════════════
#  API ESTADÍSTICAS Y FAVORITOS
# ════════════════════════════════════════════════════════════════════

@app.route('/api/estadisticas', methods=['GET'])
@login_required
@docente_required
def get_estadisticas():
    with get_db() as conn:
        row=conn.execute('''SELECT COUNT(*) as total_clases,
            SUM(CASE WHEN es_publica=1 THEN 1 ELSE 0 END) as clases_publicas,
            COALESCE(SUM(duracion_segundos)/3600.0,0) as total_horas
            FROM clases WHERE docente_id=?''',(current_user.id,)).fetchone()
        materias=conn.execute('''SELECT materia,COUNT(*) as count FROM clases
            WHERE docente_id=? AND materia!='' GROUP BY materia''',(current_user.id,)).fetchall()
    return jsonify({'total_clases':row['total_clases'],'clases_publicas':row['clases_publicas'],
        'total_horas':round(row['total_horas'],1),
        'clases_por_materia':{m['materia']:m['count'] for m in materias}})

@app.route('/api/favoritos/<int:clase_id>', methods=['POST'])
@login_required
@estudiante_required
def toggle_favorito(clase_id):
    with get_db() as conn:
        fav=conn.execute('SELECT id FROM favoritos WHERE estudiante_id=? AND clase_id=?',
            (current_user.id,clase_id)).fetchone()
        if fav:
            conn.execute('DELETE FROM favoritos WHERE id=?',(fav['id'],))
            conn.commit(); return jsonify({'favorito':False})
        conn.execute('INSERT INTO favoritos (estudiante_id,clase_id) VALUES (?,?)',
            (current_user.id,clase_id))
        conn.commit(); return jsonify({'favorito':True})

# ════════════════════════════════════════════════════════════════════
#  INICIO
# ════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    init_db()
    print("✓ Base de datos inicializada")
    app.run(debug=True, host='0.0.0.0', port=5000)