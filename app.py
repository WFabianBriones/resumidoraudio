import os
import whisper
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from datetime import datetime, timedelta
import json
import subprocess
import sqlite3
from contextlib import contextmanager
from functools import wraps

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tu-clave-secreta-cambiala-en-produccion'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

CORS(app)

UPLOAD_FOLDER = 'uploads'
DATABASE = 'clases.db'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'm4a', 'ogg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ─── Flask-Login ────────────────────────────────────────────────────────────

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, email, nombre, rol):
        self.id = id
        self.email = email
        self.nombre = nombre
        self.rol = rol

@login_manager.user_loader
def load_user(user_id):
    with get_db() as conn:
        user = conn.execute('SELECT * FROM usuarios WHERE id = ?', (user_id,)).fetchone()
        if user:
            return User(user['id'], user['email'], user['nombre'], user['rol'])
    return None

# ─── Decoradores de rol ──────────────────────────────────────────────────────

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

# ─── Base de datos ───────────────────────────────────────────────────────────

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
        conn.execute('''
            CREATE TABLE IF NOT EXISTS usuarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                nombre TEXT NOT NULL,
                rol TEXT DEFAULT 'docente',        -- 'docente' | 'estudiante'
                institucion TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS clases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                docente_id INTEGER NOT NULL,
                titulo TEXT NOT NULL,
                fecha_grabacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                duracion_segundos INTEGER,
                archivo_audio TEXT,
                transcripcion TEXT,
                resumen TEXT,
                materia TEXT,
                notas TEXT,
                es_publica BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (docente_id) REFERENCES usuarios(id) ON DELETE CASCADE
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                clase_id INTEGER,
                tag TEXT,
                FOREIGN KEY (clase_id) REFERENCES clases(id) ON DELETE CASCADE
            )
        ''')
        # Favoritos de estudiantes
        conn.execute('''
            CREATE TABLE IF NOT EXISTS favoritos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                estudiante_id INTEGER NOT NULL,
                clase_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(estudiante_id, clase_id),
                FOREIGN KEY (estudiante_id) REFERENCES usuarios(id) ON DELETE CASCADE,
                FOREIGN KEY (clase_id) REFERENCES clases(id) ON DELETE CASCADE
            )
        ''')
        conn.commit()

# ─── Cargar Whisper ──────────────────────────────────────────────────────────

print("Cargando modelo Whisper base en CPU...")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
whisper_model = whisper.load_model("base", device="cpu")
print("Whisper cargado correctamente")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ─── Resumen con Llama ───────────────────────────────────────────────────────

def generate_summary(text):
    """Genera resumen con Llama usando el método analítico-sintético."""
    prompt = f"""Eres un asistente educativo. Analiza la siguiente transcripción de una clase universitaria y genera un resumen estructurado aplicando el método analítico-sintético.

El resumen debe seguir exactamente esta estructura:
1. INTRODUCCIÓN: Tema principal de la clase
2. CONCEPTOS FUNDAMENTALES: Definiciones y teorías clave
3. DESARROLLO: Explicación de los temas tratados
4. EJEMPLOS Y APLICACIONES: Casos prácticos mencionados
5. CONCLUSIONES: Ideas principales para recordar

Transcripción:
{text[:4000]}

Resumen estructurado:"""

    try:
        result = subprocess.run(
            ['ollama', 'run', 'llama3.1:8b', prompt],
            capture_output=True, text=True, timeout=300
        )
        return result.stdout.strip() or "No se pudo generar el resumen."
    except Exception as e:
        return f"Error al generar resumen: {str(e)}"

# ════════════════════════════════════════════════════════════════════
#   RUTAS — PÁGINAS
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
            SELECT
                COUNT(*) as total_clases,
                SUM(CASE WHEN es_publica=1 THEN 1 ELSE 0 END) as clases_publicas,
                COALESCE(SUM(duracion_segundos) / 3600.0, 0) as total_horas,
                COUNT(DISTINCT materia) as total_materias
            FROM clases WHERE docente_id = ?
        ''', (current_user.id,)).fetchone()

        clases_recientes = conn.execute('''
            SELECT id, titulo, materia, es_publica, fecha_grabacion
            FROM clases WHERE docente_id = ?
            ORDER BY fecha_grabacion DESC LIMIT 10
        ''', (current_user.id,)).fetchall()

    return render_template('docente.html',
        nombre=current_user.nombre,
        total_clases=stats['total_clases'],
        clases_publicas=stats['clases_publicas'],
        total_horas=stats['total_horas'],
        total_materias=stats['total_materias'],
        clases_recientes=clases_recientes
    )

@app.route('/estudiante')
@login_required
def estudiante_dashboard():
    if current_user.rol != 'estudiante':
        return redirect(url_for('docente_dashboard'))

    with get_db() as conn:
        favoritos = conn.execute(
            'SELECT COUNT(*) as n FROM favoritos WHERE estudiante_id = ?',
            (current_user.id,)
        ).fetchone()['n']

    return render_template('estudiante.html',
        nombre=current_user.nombre,
        clases_vistas=0,
        favoritos=favoritos,
        materias=0
    )

# Ruta demo (acceso directo sin contraseña, solo para desarrollo)
@app.route('/demo/<rol>')
def demo_login(rol):
    if rol not in ('docente', 'estudiante'):
        return redirect(url_for('home'))

    demo_email = f'demo_{rol}@edutranscribe.local'
    with get_db() as conn:
        user = conn.execute('SELECT * FROM usuarios WHERE email = ?', (demo_email,)).fetchone()
        if not user:
            conn.execute(
                'INSERT INTO usuarios (email, password_hash, nombre, rol) VALUES (?, ?, ?, ?)',
                (demo_email, generate_password_hash('demo1234'), f'Demo {rol.capitalize()}', rol)
            )
            conn.commit()
            user = conn.execute('SELECT * FROM usuarios WHERE email = ?', (demo_email,)).fetchone()

    user_obj = User(user['id'], user['email'], user['nombre'], user['rol'])
    login_user(user_obj, remember=False)

    if rol == 'docente':
        return redirect(url_for('docente_dashboard'))
    return redirect(url_for('estudiante_dashboard'))

# ════════════════════════════════════════════════════════════════════
#   RUTAS — API
# ════════════════════════════════════════════════════════════════════

@app.route('/api/registro', methods=['POST'])
def api_registro():
    data = request.get_json()
    nombre = data.get('nombre', '').strip()
    email  = data.get('email', '').strip()
    password = data.get('password', '')
    institucion = data.get('institucion', '')
    rol = data.get('rol', 'estudiante')   # 'docente' | 'estudiante'

    if not nombre or not email or not password:
        return jsonify({'error': 'Datos incompletos'}), 400
    if len(password) < 8:
        return jsonify({'error': 'La contraseña debe tener al menos 8 caracteres'}), 400
    if rol not in ('docente', 'estudiante'):
        return jsonify({'error': 'Rol inválido'}), 400

    try:
        with get_db() as conn:
            conn.execute(
                'INSERT INTO usuarios (email, password_hash, nombre, institucion, rol) VALUES (?, ?, ?, ?, ?)',
                (email, generate_password_hash(password), nombre, institucion, rol)
            )
            conn.commit()
            user = conn.execute('SELECT * FROM usuarios WHERE email = ?', (email,)).fetchone()

        user_obj = User(user['id'], user['email'], user['nombre'], user['rol'])
        login_user(user_obj, remember=True)

        redirect_url = url_for('docente_dashboard') if rol == 'docente' else url_for('estudiante_dashboard')
        return jsonify({'success': True, 'redirect': redirect_url})

    except sqlite3.IntegrityError:
        return jsonify({'error': 'El correo ya está registrado'}), 400

@app.route('/api/login', methods=['POST'])
def api_login():
    # Acepta form data (del formulario HTML con JS fetch)
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form

    email    = data.get('email', '').strip()
    password = data.get('password', '')

    with get_db() as conn:
        user = conn.execute('SELECT * FROM usuarios WHERE email = ?', (email,)).fetchone()

    if user and check_password_hash(user['password_hash'], password):
        user_obj = User(user['id'], user['email'], user['nombre'], user['rol'])
        login_user(user_obj, remember=True)

        redirect_url = url_for('docente_dashboard') if user['rol'] == 'docente' else url_for('estudiante_dashboard')
        return jsonify({'success': True, 'redirect': redirect_url, 'nombre': user['nombre']})

    return jsonify({'error': 'Correo o contraseña incorrectos'}), 401

@app.route('/api/logout', methods=['POST'])
@login_required
def api_logout():
    logout_user()
    return redirect(url_for('home'))

# ── Clases (docente) ──────────────────────────────────────────────────────────

@app.route('/api/upload', methods=['POST'])
@login_required
@docente_required
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No se recibió archivo de audio'}), 400

    file = request.files['audio']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Formato de archivo no permitido'}), 400

    titulo     = request.form.get('titulo', 'Clase sin título')
    materia    = request.form.get('materia', '')
    tags       = request.form.get('tags', '')
    notas      = request.form.get('notas', '')
    es_publica = request.form.get('es_publica', 'true').lower() == 'true'

    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        print(f"Transcribiendo: {filepath}")
        result = whisper_model.transcribe(filepath, language='es')
        transcripcion = result['text']

        print("Generando resumen con Llama...")
        resumen = generate_summary(transcripcion)

        duracion = int(result.get('duration', 0))

        with get_db() as conn:
            cursor = conn.execute('''
                INSERT INTO clases (docente_id, titulo, duracion_segundos, archivo_audio,
                                    transcripcion, resumen, materia, notas, es_publica)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (current_user.id, titulo, duracion, filepath,
                  transcripcion, resumen, materia, notas, es_publica))
            clase_id = cursor.lastrowid

            if tags:
                for tag in tags.split(','):
                    tag = tag.strip()
                    if tag:
                        conn.execute('INSERT INTO tags (clase_id, tag) VALUES (?, ?)', (clase_id, tag))
            conn.commit()

        return jsonify({'success': True, 'transcript': transcripcion, 'summary': resumen, 'clase_id': clase_id})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clases', methods=['GET'])
@login_required
@docente_required
def get_clases():
    with get_db() as conn:
        clases = conn.execute('''
            SELECT id, titulo, fecha_grabacion, duracion_segundos, materia, es_publica,
                   substr(transcripcion, 1, 200) as preview_transcripcion
            FROM clases WHERE docente_id = ?
            ORDER BY fecha_grabacion DESC
        ''', (current_user.id,)).fetchall()
    return jsonify([dict(c) for c in clases])

@app.route('/api/clases/publicas', methods=['GET'])
def get_clases_publicas():
    fecha   = request.args.get('fecha')
    docente = request.args.get('docente')
    materia = request.args.get('materia')

    query = '''
        SELECT c.id, c.titulo, c.fecha_grabacion, c.materia,
               u.nombre as docente_nombre,
               substr(c.transcripcion, 1, 200) as preview_transcripcion
        FROM clases c JOIN usuarios u ON c.docente_id = u.id
        WHERE c.es_publica = 1
    '''
    params = []
    if fecha:    query += ' AND date(c.fecha_grabacion) = ?'; params.append(fecha)
    if docente:  query += ' AND u.nombre LIKE ?'; params.append(f'%{docente}%')
    if materia:  query += ' AND c.materia LIKE ?'; params.append(f'%{materia}%')
    query += ' ORDER BY c.fecha_grabacion DESC LIMIT 50'

    with get_db() as conn:
        clases = conn.execute(query, params).fetchall()
    return jsonify([dict(c) for c in clases])

@app.route('/api/clase/<int:clase_id>', methods=['GET'])
def get_clase(clase_id):
    with get_db() as conn:
        clase = conn.execute('''
            SELECT c.*, u.nombre as docente_nombre
            FROM clases c JOIN usuarios u ON c.docente_id = u.id
            WHERE c.id = ? AND (c.es_publica = 1 OR c.docente_id = ?)
        ''', (clase_id, current_user.id if current_user.is_authenticated else -1)).fetchone()

        if not clase:
            return jsonify({'error': 'Clase no encontrada o acceso denegado'}), 404

        tags = conn.execute('SELECT tag FROM tags WHERE clase_id = ?', (clase_id,)).fetchall()
        clase_dict = dict(clase)
        clase_dict['tags'] = [t['tag'] for t in tags]
        return jsonify(clase_dict)

@app.route('/api/clase/<int:clase_id>', methods=['DELETE'])
@login_required
@docente_required
def delete_clase(clase_id):
    with get_db() as conn:
        clase = conn.execute('SELECT archivo_audio, docente_id FROM clases WHERE id = ?', (clase_id,)).fetchone()
        if not clase:
            return jsonify({'error': 'Clase no encontrada'}), 404
        if clase['docente_id'] != current_user.id:
            return jsonify({'error': 'Sin permiso'}), 403
        if clase['archivo_audio']:
            try: os.remove(clase['archivo_audio'])
            except: pass
        conn.execute('DELETE FROM clases WHERE id = ?', (clase_id,))
        conn.execute('DELETE FROM tags WHERE clase_id = ?', (clase_id,))
        conn.commit()
    return jsonify({'success': True})

@app.route('/api/clase/<int:clase_id>/toggle-public', methods=['POST'])
@login_required
@docente_required
def toggle_public(clase_id):
    with get_db() as conn:
        clase = conn.execute('SELECT docente_id, es_publica FROM clases WHERE id = ?', (clase_id,)).fetchone()
        if not clase or clase['docente_id'] != current_user.id:
            return jsonify({'error': 'No autorizado'}), 403
        nuevo = not clase['es_publica']
        conn.execute('UPDATE clases SET es_publica = ? WHERE id = ?', (nuevo, clase_id))
        conn.commit()
    return jsonify({'success': True, 'es_publica': nuevo})

# ── Favoritos (estudiante) ────────────────────────────────────────────────────

@app.route('/api/favoritos/<int:clase_id>', methods=['POST'])
@login_required
@estudiante_required
def toggle_favorito(clase_id):
    with get_db() as conn:
        fav = conn.execute(
            'SELECT id FROM favoritos WHERE estudiante_id = ? AND clase_id = ?',
            (current_user.id, clase_id)
        ).fetchone()
        if fav:
            conn.execute('DELETE FROM favoritos WHERE id = ?', (fav['id'],))
            conn.commit()
            return jsonify({'favorito': False})
        else:
            conn.execute('INSERT INTO favoritos (estudiante_id, clase_id) VALUES (?, ?)',
                         (current_user.id, clase_id))
            conn.commit()
            return jsonify({'favorito': True})

# ── Estadísticas (docente) ────────────────────────────────────────────────────

@app.route('/api/estadisticas', methods=['GET'])
@login_required
@docente_required
def get_estadisticas():
    with get_db() as conn:
        row = conn.execute('''
            SELECT COUNT(*) as total_clases,
                   SUM(CASE WHEN es_publica=1 THEN 1 ELSE 0 END) as clases_publicas,
                   COALESCE(SUM(duracion_segundos)/3600.0, 0) as total_horas
            FROM clases WHERE docente_id = ?
        ''', (current_user.id,)).fetchone()

        materias = conn.execute('''
            SELECT materia, COUNT(*) as count
            FROM clases WHERE docente_id = ? AND materia != ''
            GROUP BY materia
        ''', (current_user.id,)).fetchall()

    return jsonify({
        'total_clases':   row['total_clases'],
        'clases_publicas': row['clases_publicas'],
        'total_horas':    round(row['total_horas'], 1),
        'clases_por_materia': {m['materia']: m['count'] for m in materias}
    })

# ════════════════════════════════════════════════════════════════════
#   INICIO
# ════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    init_db()
    print("Base de datos inicializada")
    app.run(debug=True, host='0.0.0.0', port=5000)