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
import sqlite3
from contextlib import contextmanager
from functools import wraps

# ════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════

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

# ════════════════════════════════════════════════════════════════════
#  FLASK-LOGIN
# ════════════════════════════════════════════════════════════════════

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

# ════════════════════════════════════════════════════════════════════
#  DECORADORES DE ROL
# ════════════════════════════════════════════════════════════════════

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

# ════════════════════════════════════════════════════════════════════
#  BASE DE DATOS
# ════════════════════════════════════════════════════════════════════

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
        conn.commit()

# ════════════════════════════════════════════════════════════════════
#  WHISPER
# ════════════════════════════════════════════════════════════════════

print("Cargando Whisper base en CPU...")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
whisper_model = whisper.load_model("base", device="cpu")
print("Whisper listo")

def allowed_file(f):
    return '.' in f and f.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ════════════════════════════════════════════════════════════════════
#  PROCESAMIENTO DE AUDIO
# ════════════════════════════════════════════════════════════════════

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
    corrected = text
    fixes_made = []
    for err, fix in fixes.items():
        if err in corrected:
            corrected = corrected.replace(err, fix)
            fixes_made.append(f"{err.strip()} → {fix.strip()}")
    if fixes_made:
        print(f"  ✓ Correcciones automáticas: {len(fixes_made)}")
        for f in fixes_made[:3]:
            print(f"    • {f}")
    return corrected

def transcribe_audio(audio_path):
    """Transcribe audio con Whisper y aplica post-procesamiento."""
    print(f"🎤 Transcribiendo: {audio_path}")
    result = whisper_model.transcribe(audio_path, language='es', verbose=False, fp16=False)
    transcript = result['text']
    print(f"✓ Transcripción base: {len(transcript)} caracteres")
    return post_process_transcript(transcript)

def chunk_text(text, chunk_size=5000):
    """Divide texto en chunks para Llama."""
    words = text.split()
    chunks, current, length = [], [], 0
    for word in words:
        length += len(word) + 1
        if length > chunk_size:
            chunks.append(' '.join(current))
            current, length = [word], len(word)
        else:
            current.append(word)
    if current:
        chunks.append(' '.join(current))
    return chunks

def validate_summary(summary, chunk_original=""):
    """Detecta posibles alucinaciones en resúmenes de IA."""
    errors_found = []
    summary_lower = summary.lower()

    hallucination_phrases = [
        'como sabemos', 'es bien conocido que',
        'históricamente se sabe', 'según los expertos',
    ]
    for phrase in hallucination_phrases:
        if phrase in summary_lower and (not chunk_original or phrase not in chunk_original.lower()):
            errors_found.append(f"⚠️ Posible alucinación: '{phrase}'")

    summary_years  = set(re.findall(r'\b(1[0-9]{3}|2[0-9]{3})\b', summary))
    if chunk_original:
        original_years = set(re.findall(r'\b(1[0-9]{3}|2[0-9]{3})\b', chunk_original))
        invented = summary_years - original_years
        if invented:
            errors_found.append(f"⚠️ Fechas no presentes en original: {invented}")

    if errors_found:
        print(f"  🔍 Validación: {len(errors_found)} advertencias")
        for e in errors_found[:5]:
            print(f"    {e}")
    return summary

def summarize_with_llama(text):
    """Genera resumen completo con Llama 3.1-8B usando método analítico-sintético."""
    chunks    = chunk_text(text, chunk_size=5000)
    summaries = []

    print(f"Procesando {len(chunks)} chunk(s) con Llama 3.1-8B...")

    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}/{len(chunks)}...")
        prompt = f"""Eres un asistente académico experto. Crea un resumen PRECISO del siguiente fragmento.

REGLAS CRÍTICAS:
- Basa tu resumen ÚNICAMENTE en la información presente en el texto
- NO agregues información externa, fechas o eventos no mencionados
- Mantén nombres propios, lugares y fechas EXACTAMENTE como aparecen

Fragmento de clase:
{chunk}

Proporciona (basándote SOLO en el texto):
1. Conceptos principales explicados
2. Puntos clave y definiciones importantes
3. Ejemplos o casos mencionados
4. Relaciones entre conceptos

Resumen académico:"""
        try:
            result  = subprocess.run(
                ['ollama', 'run', 'llama3.1:8b', prompt],
                capture_output=True, text=True, timeout=400
            )
            summary = validate_summary(result.stdout.strip(), chunk)
            summaries.append(summary)
        except subprocess.TimeoutExpired:
            summaries.append(f"[Timeout en chunk {i+1}]")
        except Exception as e:
            summaries.append(f"[Error en chunk {i+1}: {str(e)}]")

    if len(summaries) == 1:
        return summaries[0]

    # Consolidar múltiples chunks
    final_summary = "\n\n".join([f"PARTE {i+1}\n\n{s}" for i, s in enumerate(summaries)])
    consolidation_prompt = f"""Eres un asistente académico experto. Crea un resumen consolidado.

REGLAS CRÍTICAS:
- Usa SOLO la información de los resúmenes proporcionados
- NO agregues información externa
- Mantén fechas y nombres EXACTAMENTE como aparecen

Resúmenes parciales:
{final_summary}

Resumen final consolidado:
1. INTRODUCCIÓN: Tema principal
2. CONCEPTOS FUNDAMENTALES: Definiciones y teorías clave
3. DESARROLLO: Explicación de los temas
4. EJEMPLOS Y APLICACIONES: Casos prácticos
5. CONCLUSIONES: Ideas principales

IMPORTANTE: Basa el resumen SOLO en los fragmentos proporcionados.

Resumen consolidado:"""

    try:
        result = subprocess.run(
            ['ollama', 'run', 'llama3.1:8b', consolidation_prompt],
            capture_output=True, text=True, timeout=400
        )
        consolidated = validate_summary(result.stdout.strip())
        print("✓ Resumen consolidado completado")
        return f"{final_summary}\n\n{'='*50}\n\nRESUMEN CONSOLIDADO\n{'='*50}\n\n{consolidated}"
    except Exception as e:
        print(f"⚠️ Error en consolidación: {str(e)}")
        return final_summary

def generate_summary(text):
    """Wrapper para resumir texto (usado en upload completo)."""
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

@app.route('/docente')
@login_required
def docente_dashboard():
    if current_user.rol != 'docente':
        return redirect(url_for('estudiante_dashboard'))
    with get_db() as conn:
        stats = conn.execute('''
            SELECT COUNT(*) as total_clases,
                SUM(CASE WHEN es_publica=1 THEN 1 ELSE 0 END) as clases_publicas,
                COALESCE(SUM(duracion_segundos)/3600.0,0) as total_horas,
                COUNT(DISTINCT NULLIF(materia,'')) as total_materias
            FROM clases WHERE docente_id=?
        ''', (current_user.id,)).fetchone()
        clases_recientes = conn.execute('''
            SELECT id,titulo,materia,es_publica,fecha_grabacion
            FROM clases WHERE docente_id=?
            ORDER BY fecha_grabacion DESC LIMIT 10
        ''', (current_user.id,)).fetchall()
    return render_template('docente.html',
        nombre=current_user.nombre,
        total_clases=stats['total_clases'],
        clases_publicas=stats['clases_publicas'],
        total_horas=stats['total_horas'],
        total_materias=stats['total_materias'],
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
        stats = conn.execute('''
            SELECT COUNT(*) as total_clases,
                SUM(CASE WHEN es_publica=1 THEN 1 ELSE 0 END) as clases_publicas,
                COALESCE(SUM(duracion_segundos)/3600.0,0) as total_horas,
                COUNT(DISTINCT NULLIF(materia,'')) as total_materias
            FROM clases WHERE docente_id=?
        ''', (current_user.id,)).fetchone()
        por_materia = conn.execute('''
            SELECT materia, COUNT(*) as total
            FROM clases WHERE docente_id=? AND materia!=''
            GROUP BY materia ORDER BY total DESC
        ''', (current_user.id,)).fetchall()
        recientes = conn.execute('''
            SELECT titulo, fecha_grabacion, duracion_segundos, es_publica
            FROM clases WHERE docente_id=?
            ORDER BY fecha_grabacion DESC LIMIT 5
        ''', (current_user.id,)).fetchall()
    return render_template('docente_estadisticas.html',
        nombre=current_user.nombre,
        total_clases=stats['total_clases'],
        clases_publicas=stats['clases_publicas'],
        total_horas=stats['total_horas'],
        total_materias=stats['total_materias'],
        por_materia=por_materia,
        recientes=recientes)

@app.route('/docente/perfil')
@login_required
def docente_perfil():
    if current_user.rol != 'docente':
        return redirect(url_for('estudiante_dashboard'))
    with get_db() as conn:
        user = conn.execute('SELECT * FROM usuarios WHERE id=?', (current_user.id,)).fetchone()
    return render_template('docente_perfil.html',
        nombre=current_user.nombre,
        email=current_user.email,
        institucion=user['institucion'] or '')

@app.route('/estudiante')
@login_required
def estudiante_dashboard():
    if current_user.rol != 'estudiante':
        return redirect(url_for('docente_dashboard'))
    with get_db() as conn:
        fav = conn.execute(
            'SELECT COUNT(*) as n FROM favoritos WHERE estudiante_id=?',
            (current_user.id,)
        ).fetchone()['n']
    return render_template('estudiante.html',
        nombre=current_user.nombre, clases_vistas=0, favoritos=fav, materias=0)

@app.route('/demo/<rol>')
def demo_login(rol):
    if rol not in ('docente', 'estudiante'):
        return redirect(url_for('home'))
    email = f'demo_{rol}@edutranscribe.local'
    with get_db() as conn:
        user = conn.execute('SELECT * FROM usuarios WHERE email=?', (email,)).fetchone()
        if not user:
            conn.execute(
                'INSERT INTO usuarios (email,password_hash,nombre,rol) VALUES (?,?,?,?)',
                (email, generate_password_hash('demo1234'), f'Demo {rol.capitalize()}', rol)
            )
            conn.commit()
            user = conn.execute('SELECT * FROM usuarios WHERE email=?', (email,)).fetchone()
    login_user(User(user['id'], user['email'], user['nombre'], user['rol']), remember=False)
    return redirect(url_for('docente_dashboard') if rol == 'docente' else url_for('estudiante_dashboard'))

# ════════════════════════════════════════════════════════════════════
#  API AUTH
# ════════════════════════════════════════════════════════════════════

@app.route('/api/registro', methods=['POST'])
def api_registro():
    d           = request.get_json()
    nombre      = d.get('nombre', '').strip()
    email       = d.get('email', '').strip()
    password    = d.get('password', '')
    institucion = d.get('institucion', '')
    rol         = d.get('rol', 'estudiante')

    if not nombre or not email or not password:
        return jsonify({'error': 'Datos incompletos'}), 400
    if len(password) < 8:
        return jsonify({'error': 'Contraseña mínimo 8 caracteres'}), 400
    if rol not in ('docente', 'estudiante'):
        return jsonify({'error': 'Rol inválido'}), 400
    try:
        with get_db() as conn:
            conn.execute(
                'INSERT INTO usuarios (email,password_hash,nombre,institucion,rol) VALUES (?,?,?,?,?)',
                (email, generate_password_hash(password), nombre, institucion, rol)
            )
            conn.commit()
            user = conn.execute('SELECT * FROM usuarios WHERE email=?', (email,)).fetchone()
        login_user(User(user['id'], user['email'], user['nombre'], user['rol']), remember=True)
        return jsonify({
            'success':  True,
            'redirect': url_for('docente_dashboard') if rol == 'docente' else url_for('estudiante_dashboard')
        })
    except sqlite3.IntegrityError:
        return jsonify({'error': 'El correo ya está registrado'}), 400

@app.route('/api/login', methods=['POST'])
def api_login():
    d        = request.get_json() if request.is_json else request.form
    email    = d.get('email', '').strip()
    password = d.get('password', '')

    with get_db() as conn:
        user = conn.execute('SELECT * FROM usuarios WHERE email=?', (email,)).fetchone()

    if user and check_password_hash(user['password_hash'], password):
        login_user(User(user['id'], user['email'], user['nombre'], user['rol']), remember=True)
        return jsonify({
            'success':  True,
            'nombre':   user['nombre'],
            'redirect': url_for('docente_dashboard') if user['rol'] == 'docente' else url_for('estudiante_dashboard')
        })
    return jsonify({'error': 'Correo o contraseña incorrectos'}), 401

@app.route('/api/logout', methods=['POST'])
@login_required
def api_logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/api/perfil', methods=['POST'])
@login_required
def api_actualizar_perfil():
    data        = request.get_json()
    nombre      = data.get('nombre', '').strip()
    institucion = data.get('institucion', '').strip()
    password    = data.get('password', '').strip()

    if not nombre:
        return jsonify({'error': 'El nombre no puede estar vacío'}), 400

    with get_db() as conn:
        if password:
            if len(password) < 8:
                return jsonify({'error': 'La contraseña debe tener al menos 8 caracteres'}), 400
            conn.execute(
                'UPDATE usuarios SET nombre=?, institucion=?, password_hash=? WHERE id=?',
                (nombre, institucion, generate_password_hash(password), current_user.id)
            )
        else:
            conn.execute(
                'UPDATE usuarios SET nombre=?, institucion=? WHERE id=?',
                (nombre, institucion, current_user.id)
            )
        conn.commit()
    return jsonify({'success': True})

# ════════════════════════════════════════════════════════════════════
#  API TRANSCRIPCIÓN EN TIEMPO REAL
# ════════════════════════════════════════════════════════════════════

@app.route('/api/transcribe-chunk', methods=['POST'])
@login_required
@docente_required
def transcribe_chunk():
    """Recibe blob ~8s de MediaRecorder, transcribe con Whisper y devuelve texto."""
    if 'chunk' not in request.files:
        return jsonify({'error': 'No se recibió chunk de audio'}), 400

    chunk_file = request.files['chunk']
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
        tmp_path = tmp.name
        chunk_file.save(tmp_path)

    try:
        result = whisper_model.transcribe(tmp_path, language='es', fp16=False)
        text   = post_process_transcript(result['text'].strip())
        return jsonify({'success': True, 'text': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass

# ════════════════════════════════════════════════════════════════════
#  API RESUMEN STREAMING (SSE)
# ════════════════════════════════════════════════════════════════════

@app.route('/api/summary-stream', methods=['POST'])
@login_required
@docente_required
def summary_stream():
    """Streaming del resumen token a token via SSE + Ollama REST API."""
    data          = request.get_json()
    transcripcion = data.get('text', '').strip()
    if not transcripcion:
        return jsonify({'error': 'Sin texto para resumir'}), 400

    prompt = f"""Eres un asistente educativo universitario. Analiza esta transcripción de clase y genera un resumen aplicando el método analítico-sintético.

Usa EXACTAMENTE esta estructura:

## 1. INTRODUCCIÓN
[Tema principal de la clase]

## 2. CONCEPTOS FUNDAMENTALES
[Definiciones y teorías clave mencionadas]

## 3. DESARROLLO
[Explicación detallada de los temas tratados]

## 4. EJEMPLOS Y APLICACIONES
[Casos prácticos y ejemplos mencionados]

## 5. CONCLUSIONES
[Ideas principales para recordar y estudiar]

Transcripción:
{transcripcion[:6000]}

Resumen:"""

    def generate_sse():
        try:
            response = req_lib.post(
                'http://localhost:11434/api/generate',
                json={
                    'model':   'llama3.1:8b',
                    'prompt':  prompt,
                    'stream':  True,
                    'options': {'temperature': 0.3, 'num_predict': 1200}
                },
                stream=True, timeout=300
            )
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        token = chunk.get('response', '')
                        if token:
                            yield f"data: {json.dumps({'token': token})}\n\n"
                        if chunk.get('done'):
                            yield "data: [DONE]\n\n"
                            return
                    except json.JSONDecodeError:
                        continue
        except Exception:
            yield f"data: {json.dumps({'token': 'Generando resumen...\n\n'})}\n\n"
            try:
                r    = subprocess.run(
                    ['ollama', 'run', 'llama3.1:8b', prompt],
                    capture_output=True, text=True, timeout=300
                )
                full = r.stdout.strip()
                for i in range(0, len(full), 6):
                    yield f"data: {json.dumps({'token': full[i:i+6]})}\n\n"
            except Exception as e2:
                yield f"data: {json.dumps({'error': str(e2)})}\n\n"
            yield "data: [DONE]\n\n"

    return Response(
        generate_sse(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control':     'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection':        'keep-alive',
        }
    )

# ════════════════════════════════════════════════════════════════════
#  API GUARDAR CLASE (transcripción + resumen ya listos del frontend)
# ════════════════════════════════════════════════════════════════════

@app.route('/api/guardar-clase', methods=['POST'])
@login_required
@docente_required
def guardar_clase():
    d             = request.get_json()
    titulo        = d.get('titulo', 'Clase sin título').strip()
    materia       = d.get('materia', '').strip()
    tags_str      = d.get('tags', '').strip()
    notas         = d.get('notas', '').strip()
    transcripcion = d.get('transcripcion', '').strip()
    resumen       = d.get('resumen', '').strip()
    es_publica    = d.get('es_publica', True)
    duracion      = d.get('duracion_segundos', 0)

    if not transcripcion:
        return jsonify({'error': 'Sin transcripción'}), 400

    with get_db() as conn:
        cur = conn.execute('''
            INSERT INTO clases
              (docente_id,titulo,duracion_segundos,transcripcion,resumen,materia,notas,es_publica)
            VALUES (?,?,?,?,?,?,?,?)
        ''', (current_user.id, titulo, duracion, transcripcion, resumen, materia, notas, es_publica))
        cid = cur.lastrowid
        for tag in tags_str.split(','):
            tag = tag.strip()
            if tag:
                conn.execute('INSERT INTO tags (clase_id,tag) VALUES (?,?)', (cid, tag))
        conn.commit()
    return jsonify({'success': True, 'clase_id': cid})

# ════════════════════════════════════════════════════════════════════
#  API UPLOAD (archivo completo: transcribe + resume + guarda)
# ════════════════════════════════════════════════════════════════════

@app.route('/api/upload', methods=['POST'])
@login_required
@docente_required
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'Sin archivo'}), 400
    file = request.files['audio']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Formato no permitido'}), 400

    titulo     = request.form.get('titulo', 'Clase sin título')
    materia    = request.form.get('materia', '')
    tags       = request.form.get('tags', '')
    notas      = request.form.get('notas', '')
    es_publica = request.form.get('es_publica', 'true').lower() == 'true'

    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        print(f"📁 Archivo recibido: {file.filename}")
        result    = whisper_model.transcribe(filepath, language='es', fp16=False)
        tr        = post_process_transcript(result['text'])
        dur       = int(result.get('duration', 0))

        print("🤖 Generando resumen con Llama...")
        resumen = summarize_with_llama(tr)
        print(f"✓ Resumen: {len(resumen)} caracteres")

        with get_db() as conn:
            cur = conn.execute('''
                INSERT INTO clases
                  (docente_id,titulo,duracion_segundos,archivo_audio,
                   transcripcion,resumen,materia,notas,es_publica)
                VALUES (?,?,?,?,?,?,?,?,?)
            ''', (current_user.id, titulo, dur, filepath, tr, resumen, materia, notas, es_publica))
            cid = cur.lastrowid
            for tag in tags.split(','):
                tag = tag.strip()
                if tag:
                    conn.execute('INSERT INTO tags (clase_id,tag) VALUES (?,?)', (cid, tag))
            conn.commit()

        return jsonify({'success': True, 'transcript': tr, 'summary': resumen, 'clase_id': cid})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ════════════════════════════════════════════════════════════════════
#  API CLASES
# ════════════════════════════════════════════════════════════════════

@app.route('/api/clases', methods=['GET'])
@login_required
@docente_required
def get_clases():
    with get_db() as conn:
        clases = conn.execute('''
            SELECT id,titulo,fecha_grabacion,duracion_segundos,materia,es_publica,
                   substr(transcripcion,1,200) as preview_transcripcion,
                   substr(resumen,1,200) as preview_resumen
            FROM clases WHERE docente_id=?
            ORDER BY fecha_grabacion DESC
        ''', (current_user.id,)).fetchall()
    return jsonify([dict(c) for c in clases])

@app.route('/api/clases/publicas', methods=['GET'])
def get_clases_publicas():
    fecha   = request.args.get('fecha')
    docente = request.args.get('docente')
    materia = request.args.get('materia')

    q  = '''SELECT c.id,c.titulo,c.fecha_grabacion,c.materia,u.nombre as docente_nombre,
               substr(c.transcripcion,1,200) as preview_transcripcion
            FROM clases c JOIN usuarios u ON c.docente_id=u.id
            WHERE c.es_publica=1'''
    p  = []
    if fecha:   q += ' AND date(c.fecha_grabacion)=?'; p.append(fecha)
    if docente: q += ' AND u.nombre LIKE ?';           p.append(f'%{docente}%')
    if materia: q += ' AND c.materia LIKE ?';          p.append(f'%{materia}%')
    q += ' ORDER BY c.fecha_grabacion DESC LIMIT 50'

    with get_db() as conn:
        clases = conn.execute(q, p).fetchall()
    return jsonify([dict(c) for c in clases])

@app.route('/api/clase/<int:clase_id>', methods=['GET'])
def get_clase(clase_id):
    with get_db() as conn:
        clase = conn.execute('''
            SELECT c.*,u.nombre as docente_nombre
            FROM clases c JOIN usuarios u ON c.docente_id=u.id
            WHERE c.id=? AND (c.es_publica=1 OR c.docente_id=?)
        ''', (clase_id, current_user.id if current_user.is_authenticated else -1)).fetchone()
        if not clase:
            return jsonify({'error': 'No encontrada o acceso denegado'}), 404
        tags = conn.execute('SELECT tag FROM tags WHERE clase_id=?', (clase_id,)).fetchall()
        d = dict(clase)
        d['tags'] = [t['tag'] for t in tags]
        return jsonify(d)

@app.route('/api/clase/<int:clase_id>', methods=['DELETE'])
@login_required
@docente_required
def delete_clase(clase_id):
    with get_db() as conn:
        clase = conn.execute('SELECT archivo_audio,docente_id FROM clases WHERE id=?', (clase_id,)).fetchone()
        if not clase:
            return jsonify({'error': 'No encontrada'}), 404
        if clase['docente_id'] != current_user.id:
            return jsonify({'error': 'Sin permiso'}), 403
        if clase['archivo_audio']:
            try:
                os.remove(clase['archivo_audio'])
            except:
                pass
        conn.execute('DELETE FROM clases WHERE id=?', (clase_id,))
        conn.execute('DELETE FROM tags WHERE clase_id=?', (clase_id,))
        conn.commit()
    return jsonify({'success': True})

@app.route('/api/clase/<int:clase_id>/toggle-public', methods=['POST'])
@login_required
@docente_required
def toggle_public(clase_id):
    with get_db() as conn:
        clase = conn.execute('SELECT docente_id,es_publica FROM clases WHERE id=?', (clase_id,)).fetchone()
        if not clase or clase['docente_id'] != current_user.id:
            return jsonify({'error': 'No autorizado'}), 403
        nuevo = not clase['es_publica']
        conn.execute('UPDATE clases SET es_publica=? WHERE id=?', (nuevo, clase_id))
        conn.commit()
    return jsonify({'success': True, 'es_publica': nuevo})

# ════════════════════════════════════════════════════════════════════
#  API ESTADÍSTICAS Y FAVORITOS
# ════════════════════════════════════════════════════════════════════

@app.route('/api/estadisticas', methods=['GET'])
@login_required
@docente_required
def get_estadisticas():
    with get_db() as conn:
        row = conn.execute('''
            SELECT COUNT(*) as total_clases,
                SUM(CASE WHEN es_publica=1 THEN 1 ELSE 0 END) as clases_publicas,
                COALESCE(SUM(duracion_segundos)/3600.0,0) as total_horas
            FROM clases WHERE docente_id=?
        ''', (current_user.id,)).fetchone()
        materias = conn.execute('''
            SELECT materia,COUNT(*) as count FROM clases
            WHERE docente_id=? AND materia!='' GROUP BY materia
        ''', (current_user.id,)).fetchall()
    return jsonify({
        'total_clases':       row['total_clases'],
        'clases_publicas':    row['clases_publicas'],
        'total_horas':        round(row['total_horas'], 1),
        'clases_por_materia': {m['materia']: m['count'] for m in materias}
    })

@app.route('/api/favoritos/<int:clase_id>', methods=['POST'])
@login_required
@estudiante_required
def toggle_favorito(clase_id):
    with get_db() as conn:
        fav = conn.execute(
            'SELECT id FROM favoritos WHERE estudiante_id=? AND clase_id=?',
            (current_user.id, clase_id)
        ).fetchone()
        if fav:
            conn.execute('DELETE FROM favoritos WHERE id=?', (fav['id'],))
            conn.commit()
            return jsonify({'favorito': False})
        conn.execute('INSERT INTO favoritos (estudiante_id,clase_id) VALUES (?,?)', (current_user.id, clase_id))
        conn.commit()
        return jsonify({'favorito': True})

# ════════════════════════════════════════════════════════════════════
#  INICIO  ← siempre al final
# ════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════
# PEGAR ESTAS RUTAS EN app.py ANTES DEL if __name__ == '__main__':
# ════════════════════════════════════════════════════════════════════

# ── Visor de clase (docente preview + estudiante) ─────────────────

@app.route('/clase/<int:clase_id>')
@login_required
def ver_clase(clase_id):
    """Página de visor completo: transcripción + resumen."""
    with get_db() as conn:
        clase = conn.execute('''
            SELECT c.*, u.nombre as docente_nombre
            FROM clases c JOIN usuarios u ON c.docente_id = u.id
            WHERE c.id = ? AND (c.es_publica = 1 OR c.docente_id = ?)
        ''', (clase_id, current_user.id)).fetchone()

        if not clase:
            return "Clase no encontrada o acceso denegado", 404

        tags = conn.execute(
            'SELECT tag FROM tags WHERE clase_id = ?', (clase_id,)
        ).fetchall()

        is_favorite = False
        if current_user.rol == 'estudiante':
            fav = conn.execute(
                'SELECT id FROM favoritos WHERE estudiante_id = ? AND clase_id = ?',
                (current_user.id, clase_id)
            ).fetchone()
            is_favorite = fav is not None

    clase_dict = dict(clase)
    clase_dict['tags'] = [t['tag'] for t in tags]

    es_docente = current_user.rol == 'docente' and clase['docente_id'] == current_user.id

    if es_docente:
        back_url = url_for('docente_nueva_clase') + '#biblioteca'
    else:
        back_url = url_for('estudiante_buscar')

    return render_template('clase_ver.html',
        clase       = clase_dict,
        es_docente  = es_docente,
        is_favorite = is_favorite,
        back_url    = back_url
    )

# ── Búsqueda de clases para estudiantes ──────────────────────────

@app.route('/estudiante/buscar')
@login_required
def estudiante_buscar():
    if current_user.rol != 'estudiante':
        return redirect(url_for('docente_dashboard'))
    return render_template('estudiante_buscar.html', nombre=current_user.nombre)

# ── Favoritos del estudiante ──────────────────────────────────────

@app.route('/estudiante/favoritos')
@login_required
def estudiante_favoritos():
    if current_user.rol != 'estudiante':
        return redirect(url_for('docente_dashboard'))
    with get_db() as conn:
        clases = conn.execute('''
            SELECT c.id, c.titulo, c.fecha_grabacion, c.materia,
                   u.nombre as docente_nombre
            FROM favoritos f
            JOIN clases c ON f.clase_id = c.id
            JOIN usuarios u ON c.docente_id = u.id
            WHERE f.estudiante_id = ?
            ORDER BY f.created_at DESC
        ''', (current_user.id,)).fetchall()
    return render_template('estudiante_favoritos.html',
        nombre = current_user.nombre,
        clases = clases
    )

if __name__ == '__main__':
    init_db()
    print("✓ Base de datos inicializada")
    app.run(debug=True, host='0.0.0.0', port=5000)