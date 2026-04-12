import os
import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session
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
app.config['SECRET_KEY'] = 'tu-clave-secreta-aqui-cambiala-en-produccion'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max

# Habilitar CORS para todas las rutas
CORS(app)

# Configuración
UPLOAD_FOLDER = 'uploads'
DATABASE = 'clases.db'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'm4a', 'ogg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Flask-Login
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
        user = conn.execute(
            'SELECT * FROM usuarios WHERE id = ?', (user_id,)
        ).fetchone()
        if user:
            return User(user['id'], user['email'], user['nombre'], user['rol'])
    return None

# Decorador para verificar rol de docente
def docente_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.rol != 'docente':
            return jsonify({'error': 'Acceso denegado. Solo para docentes.'}), 403
        return f(*args, **kwargs)
    return decorated_function

# Inicializar base de datos
def init_db():
    with get_db() as conn:
        # Tabla de usuarios
        conn.execute('''
            CREATE TABLE IF NOT EXISTS usuarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                nombre TEXT NOT NULL,
                rol TEXT DEFAULT 'docente',
                institucion TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabla de clases (ahora con docente_id)
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
                FOREIGN KEY (docente_id) REFERENCES usuarios (id) ON DELETE CASCADE
            )
        ''')
        
        # Tabla de tags
        conn.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                clase_id INTEGER,
                tag TEXT,
                FOREIGN KEY (clase_id) REFERENCES clases (id) ON DELETE CASCADE
            )
        ''')
        
        # Tabla de sesiones temporales para invitados
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sesiones_invitado (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE,
                transcripcion TEXT,
                resumen TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()

@contextmanager
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# Cargar modelo Whisper
print("Cargando modelo Whisper base en CPU (optimizado)...")
import torch
# Forzar uso de CPU para evitar conflictos con Ollama en GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Ocultar GPU de PyTorch
whisper_model = whisper.load_model("base", device="cpu")
print("Modelo Whisper cargado en CPU exitosamente")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def post_process_transcript(text):
    """Corrección automática GENÉRICA de errores comunes de Whisper en español"""
    
    # Solo correcciones que aplican a CUALQUIER audio en español
    generic_corrections = {
        # Palabras comúnmente mal transcritas por Whisper
        ' senstos ': ' sentidos ',
        ' exaustos ': ' exhaustos ',
        ' exausto ': ' exhausto ',
        ' amasa ': ' en masa ',
        ' cobernada ': ' gobernada ',
        ' cobernado ': ' gobernado ',
        ' mullieron ': ' murieron ',
        ' mulló ': ' murió ',
        ' pasacre ': ' masacre ',
        
        # Palabras con tildes comúnmente omitidas
        ' tambien ': ' también ',
        ' mas ': ' más ',  # Solo cuando no es "pero"
        ' si ': ' sí ',     # Contextual
        ' solo ': ' sólo ',
    }
    
    corrected_text = text
    corrections_made = []
    
    for error, correction in generic_corrections.items():
        if error in corrected_text:
            corrected_text = corrected_text.replace(error, correction)
            corrections_made.append(f"{error.strip()} → {correction.strip()}")
    
    if corrections_made:
        print(f"  ✓ Correcciones automáticas: {len(corrections_made)}")
        for corr in corrections_made[:3]:  # Mostrar primeras 3
            print(f"    • {corr}")
        if len(corrections_made) > 3:
            print(f"    • ... y {len(corrections_made) - 3} más")
    
    return corrected_text

def transcribe_audio(audio_path):
    """Transcribe audio usando Whisper con post-procesamiento genérico"""
    print(f"🎤 Iniciando transcripción de: {audio_path}")
    result = whisper_model.transcribe(
        audio_path, 
        language='es',
        verbose=False,  # Reducir output innecesario
        fp16=False      # Mejor precisión en CPU
    )
    
    # Post-procesar solo errores genéricos comunes
    transcript = result['text']
    print(f"✓ Transcripción base completada: {len(transcript)} caracteres")
    
    transcript = post_process_transcript(transcript)
    
    return transcript

def chunk_text(text, chunk_size=5000):  # Chunks más grandes = menos llamadas a Llama
    """Divide el texto en chunks para procesamiento"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_length += len(word) + 1
        if current_length > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def validate_summary(summary, chunk_original=""):
    """Detectar y advertir sobre alucinaciones comunes en resúmenes IA"""
    errors_found = []
    corrected_summary = summary
    
    # 1. Detectar anacronismos tecnológicos comunes (IA inventa tecnología moderna)
    modern_tech = {
        'computadora': 'cálculos',
        'ordenador': 'cálculos',
        'internet': 'comunicaciones',
        'celular': 'teléfono',
        'móvil': 'teléfono',
        'satélite': 'observación aérea',
        'dron': 'avión',
        'email': 'correspondencia',
        'software': 'sistema',
        'app': 'aplicación',
        'digital': 'registrado',
    }
    
    summary_lower = corrected_summary.lower()
    
    for tech_term, replacement in modern_tech.items():
        if tech_term in summary_lower:
            # Solo advertir, no corregir automáticamente (podría ser legítimo)
            errors_found.append(f"⚠️ Término moderno detectado: '{tech_term}' - Verificar contexto")
    
    # 2. Detectar frases típicas de IA alucinando
    hallucination_phrases = [
        'como sabemos',
        'es bien conocido que',
        'históricamente se sabe',
        'según los expertos',
        'la investigación muestra',
    ]
    
    for phrase in hallucination_phrases:
        if phrase in summary_lower and phrase not in chunk_original.lower():
            errors_found.append(f"⚠️ Posible alucinación: '{phrase}' no está en el texto original")
    
    # 3. Advertir si hay demasiados números/fechas no presentes en el original
    import re
    summary_years = set(re.findall(r'\b(1[0-9]{3}|2[0-9]{3})\b', corrected_summary))
    if chunk_original:
        original_years = set(re.findall(r'\b(1[0-9]{3}|2[0-9]{3})\b', chunk_original))
        invented_years = summary_years - original_years
        if invented_years:
            errors_found.append(f"⚠️ Fechas no presentes en original: {invented_years}")
    
    # 4. Mostrar advertencias
    if errors_found:
        print(f"  🔍 Validación detectó {len(errors_found)} advertencias:")
        for error in errors_found[:5]:  # Mostrar primeras 5
            print(f"    {error}")
    
    return corrected_summary

def summarize_with_llama(text):
    """Crea resumen usando Llama 3.1-8B a través de Ollama (GPU optimizado)"""
    chunks = chunk_text(text, chunk_size=5000)
    summaries = []
    
    print(f"Procesando {len(chunks)} chunks con Llama 3.1-8B...")
    
    for i, chunk in enumerate(chunks):
        print(f"Procesando chunk {i+1}/{len(chunks)}")
        
        # Prompt mejorado anti-alucinación
        prompt = f"""Eres un asistente académico experto. Tu tarea es crear un resumen PRECISO del siguiente fragmento.

REGLAS CRÍTICAS:
- Basa tu resumen ÚNICAMENTE en la información presente en el texto
- NO agregues información externa, fechas o eventos que no estén explícitamente mencionados
- NO menciones tecnologías que no aparecen en el texto (ej: NO inventes armas o tecnologías)
- Si hay fechas, VERIFICA que sean correctas en el contexto
- Mantén los nombres propios, lugares y fechas EXACTAMENTE como aparecen

Fragmento de clase:
{chunk}

Proporciona (basándote SOLO en el texto):
1. Conceptos principales explicados
2. Puntos clave y definiciones importantes
3. Ejemplos o casos mencionados específicamente
4. Relaciones entre conceptos del texto

Resumen académico (solo información del fragmento):"""

        try:
            result = subprocess.run(
                ['ollama', 'run', 'llama3.1:8b', prompt],
                capture_output=True,
                text=True,
                timeout=400  # Más tiempo para modelo más grande
            )
            summary = result.stdout.strip()
            
            # Validar y corregir el resumen
            summary = validate_summary(summary, chunk)
            
            summaries.append(summary)
        except subprocess.TimeoutExpired:
            summaries.append(f"[Timeout en chunk {i+1}]")
        except Exception as e:
            summaries.append(f"[Error en chunk {i+1}: {str(e)}]")
    
    final_summary = "\n\n--- SECCIÓN {} ---\n\n".join(
        [f"PARTE {i+1}\n\n{s}" for i, s in enumerate(summaries)]
    )
    
    # Prompt mejorado para consolidación
    consolidation_prompt = f"""Eres un asistente académico experto. Crea un resumen consolidado del material estudiado.

REGLAS CRÍTICAS:
- Usa SOLO la información de los resúmenes proporcionados
- NO agregues información externa o conocimiento general
- Mantén las fechas y nombres EXACTAMENTE como aparecen
- NO inventes eventos, tecnologías o detalles que no estén mencionados

Resúmenes parciales:
{final_summary}

Crea un resumen final consolidado que incluya:
1. INTRODUCCIÓN: Tema principal de la clase (según los resúmenes)
2. CONCEPTOS FUNDAMENTALES: Definiciones y teorías clave mencionadas
3. DESARROLLO: Explicación de los temas tratados
4. EJEMPLOS Y APLICACIONES: Casos prácticos del material
5. CONCLUSIONES: Ideas principales para recordar

IMPORTANTE: Basa el resumen SOLO en los fragmentos proporcionados.

Resumen consolidado:"""

    try:
        result = subprocess.run(
            ['ollama', 'run', 'llama3.1:8b', consolidation_prompt],
            capture_output=True,
            text=True,
            timeout=400
        )
        consolidated = result.stdout.strip()
        
        # Validar resumen consolidado
        consolidated = validate_summary(consolidated)
        
        print("✓ Resumen consolidado completado y validado")
        
        return f"{final_summary}\n\n{'='*50}\n\nRESUMEN CONSOLIDADO\n{'='*50}\n\n{consolidated}"
    except Exception as e:
        print(f"⚠️ Error en consolidación: {str(e)}")
        return final_summary

# ==================== RUTAS ====================

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/registro')
def registro():
    return render_template('registro.html')

@app.route('/api/registro', methods=['POST'])
def api_registro():
    data = request.get_json()
    
    email = data.get('email')
    password = data.get('password')
    nombre = data.get('nombre')
    institucion = data.get('institucion', '')
    
    if not email or not password or not nombre:
        return jsonify({'error': 'Datos incompletos'}), 400
    
    password_hash = generate_password_hash(password)
    
    try:
        with get_db() as conn:
            conn.execute(
                'INSERT INTO usuarios (email, password_hash, nombre, institucion, rol) VALUES (?, ?, ?, ?, ?)',
                (email, password_hash, nombre, institucion, 'docente')
            )
            conn.commit()
        return jsonify({'success': True})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'El email ya está registrado'}), 400

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    with get_db() as conn:
        user = conn.execute(
            'SELECT * FROM usuarios WHERE email = ?', (email,)
        ).fetchone()
    
    if user and check_password_hash(user['password_hash'], password):
        user_obj = User(user['id'], user['email'], user['nombre'], user['rol'])
        login_user(user_obj, remember=True)
        return jsonify({'success': True, 'nombre': user['nombre']})
    
    return jsonify({'error': 'Credenciales inválidas'}), 401

@app.route('/api/logout', methods=['POST'])
@login_required
def api_logout():
    logout_user()
    return jsonify({'success': True})

@app.route('/docente')
@login_required
@docente_required
def docente_dashboard():
    return render_template('docente.html', nombre=current_user.nombre)

@app.route('/invitado')
def invitado():
    # Crear session_id para invitado si no existe
    if 'guest_id' not in session:
        session['guest_id'] = os.urandom(16).hex()
    return render_template('invitado.html')

@app.route('/api/clases', methods=['GET'])
@login_required
@docente_required
def get_clases():
    """Obtener clases del docente actual"""
    with get_db() as conn:
        clases = conn.execute('''
            SELECT id, titulo, fecha_grabacion, duracion_segundos, 
                   materia, es_publica,
                   substr(transcripcion, 1, 200) as preview_transcripcion,
                   substr(resumen, 1, 200) as preview_resumen
            FROM clases 
            WHERE docente_id = ?
            ORDER BY fecha_grabacion DESC
        ''', (current_user.id,)).fetchall()
        
        return jsonify([dict(clase) for clase in clases])

@app.route('/api/clases/publicas', methods=['GET'])
def get_clases_publicas():
    """Obtener clases públicas (para invitados)"""
    fecha = request.args.get('fecha')
    docente = request.args.get('docente')
    materia = request.args.get('materia')
    
    query = '''
        SELECT c.id, c.titulo, c.fecha_grabacion, c.materia,
               u.nombre as docente_nombre,
               substr(c.transcripcion, 1, 200) as preview_transcripcion
        FROM clases c
        JOIN usuarios u ON c.docente_id = u.id
        WHERE c.es_publica = 1
    '''
    params = []
    
    if fecha:
        query += ' AND date(c.fecha_grabacion) = ?'
        params.append(fecha)
    if docente:
        query += ' AND u.nombre LIKE ?'
        params.append(f'%{docente}%')
    if materia:
        query += ' AND c.materia LIKE ?'
        params.append(f'%{materia}%')
    
    query += ' ORDER BY c.fecha_grabacion DESC LIMIT 50'
    
    with get_db() as conn:
        clases = conn.execute(query, params).fetchall()
        return jsonify([dict(clase) for clase in clases])

@app.route('/api/clase/<int:clase_id>', methods=['GET'])
def get_clase(clase_id):
    """Obtener una clase específica"""
    with get_db() as conn:
        # Verificar si es pública o si el usuario es el dueño
        clase = conn.execute('''
            SELECT c.*, u.nombre as docente_nombre
            FROM clases c
            JOIN usuarios u ON c.docente_id = u.id
            WHERE c.id = ? AND (c.es_publica = 1 OR c.docente_id = ?)
        ''', (clase_id, current_user.id if current_user.is_authenticated else -1)).fetchone()
        
        if clase:
            tags = conn.execute(
                'SELECT tag FROM tags WHERE clase_id = ?', (clase_id,)
            ).fetchall()
            
            clase_dict = dict(clase)
            clase_dict['tags'] = [tag['tag'] for tag in tags]
            return jsonify(clase_dict)
        
        return jsonify({'error': 'Clase no encontrada o acceso denegado'}), 404

@app.route('/api/clase/<int:clase_id>', methods=['DELETE'])
@login_required
@docente_required
def delete_clase(clase_id):
    """Eliminar una clase (solo el docente dueño)"""
    with get_db() as conn:
        clase = conn.execute(
            'SELECT archivo_audio, docente_id FROM clases WHERE id = ?', (clase_id,)
        ).fetchone()
        
        if not clase:
            return jsonify({'error': 'Clase no encontrada'}), 404
        
        if clase['docente_id'] != current_user.id:
            return jsonify({'error': 'No tienes permiso para eliminar esta clase'}), 403
        
        if clase['archivo_audio']:
            try:
                os.remove(clase['archivo_audio'])
            except:
                pass
        
        conn.execute('DELETE FROM clases WHERE id = ?', (clase_id,))
        conn.execute('DELETE FROM tags WHERE clase_id = ?', (clase_id,))
        conn.commit()
        
        return jsonify({'success': True})

@app.route('/api/clase/<int:clase_id>/toggle-public', methods=['POST'])
@login_required
@docente_required
def toggle_public(clase_id):
    """Cambiar visibilidad pública de una clase"""
    with get_db() as conn:
        clase = conn.execute(
            'SELECT docente_id, es_publica FROM clases WHERE id = ?', (clase_id,)
        ).fetchone()
        
        if not clase or clase['docente_id'] != current_user.id:
            return jsonify({'error': 'No autorizado'}), 403
        
        nuevo_estado = not clase['es_publica']
        conn.execute(
            'UPDATE clases SET es_publica = ? WHERE id = ?',
            (nuevo_estado, clase_id)
        )
        conn.commit()
        
        return jsonify({'success': True, 'es_publica': nuevo_estado})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Procesar audio - para docentes guarda, para invitados es temporal"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No se encontró archivo de audio'}), 400
        
        file = request.files['audio']
        titulo = request.form.get('titulo', 'Clase sin título')
        materia = request.form.get('materia', '')
        notas = request.form.get('notas', '')
        tags_str = request.form.get('tags', '')
        es_publica = request.form.get('es_publica', 'true') == 'true'
        
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó archivo'}), 400
        
        print(f"📁 Archivo recibido: {file.filename}")
        print(f"👤 Usuario: {'Docente' if current_user.is_authenticated else 'Invitado'}")
        
        if file and allowed_file(file.filename):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"clase_{timestamp}.wav"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            print(f"💾 Guardando en: {filepath}")
            file.save(filepath)
            
            try:
                # Transcribir audio
                print("🎤 Iniciando transcripción...")
                transcript = transcribe_audio(filepath)
                print(f"✓ Transcripción completada: {len(transcript)} caracteres")
                
                # Crear resumen
                print("🤖 Generando resumen con Llama 3...")
                summary = summarize_with_llama(transcript)
                print(f"✓ Resumen completado: {len(summary)} caracteres")
                
                # Si es docente, guardar en BD
                if current_user.is_authenticated and current_user.rol == 'docente':
                    with get_db() as conn:
                        cursor = conn.execute('''
                            INSERT INTO clases 
                            (docente_id, titulo, archivo_audio, transcripcion, resumen, materia, notas, es_publica)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (current_user.id, titulo, filepath, transcript, summary, materia, notas, es_publica))
                        
                        clase_id = cursor.lastrowid
                        
                        if tags_str:
                            tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
                            for tag in tags:
                                conn.execute(
                                    'INSERT INTO tags (clase_id, tag) VALUES (?, ?)',
                                    (clase_id, tag)
                                )
                        
                        conn.commit()
                    
                    print(f"✓ Clase guardada con ID: {clase_id}")
                    return jsonify({
                        'success': True,
                        'clase_id': clase_id,
                        'transcript': transcript,
                        'summary': summary,
                        'saved': True
                    })
                else:
                    # Invitado - no guardar, solo devolver resultado
                    print("🗑️ Eliminando archivo temporal (invitado)")
                    os.remove(filepath)
                    return jsonify({
                        'success': True,
                        'transcript': transcript,
                        'summary': summary,
                        'saved': False,
                        'message': 'Transcripción temporal (no guardada)'
                    })
            
            except Exception as e:
                print(f"❌ Error procesando audio: {str(e)}")
                import traceback
                traceback.print_exc()
                
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': f'Error procesando audio: {str(e)}'}), 500
        
        return jsonify({'error': 'Formato de archivo no permitido'}), 400
    
    except Exception as e:
        print(f"❌ Error general en upload: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error del servidor: {str(e)}'}), 500

@app.route('/api/estadisticas', methods=['GET'])
@login_required
@docente_required
def estadisticas():
    """Obtener estadísticas del docente"""
    with get_db() as conn:
        stats = {
            'total_clases': conn.execute(
                'SELECT COUNT(*) as count FROM clases WHERE docente_id = ?',
                (current_user.id,)
            ).fetchone()['count'],
            'clases_publicas': conn.execute(
                'SELECT COUNT(*) as count FROM clases WHERE docente_id = ? AND es_publica = 1',
                (current_user.id,)
            ).fetchone()['count'],
            'total_horas': conn.execute(
                'SELECT SUM(duracion_segundos) / 3600.0 as horas FROM clases WHERE docente_id = ?',
                (current_user.id,)
            ).fetchone()['horas'] or 0,
            'clases_por_materia': {}
        }
        
        materias = conn.execute('''
            SELECT materia, COUNT(*) as count 
            FROM clases 
            WHERE docente_id = ? AND materia != ''
            GROUP BY materia
        ''', (current_user.id,)).fetchall()
        
        for m in materias:
            stats['clases_por_materia'][m['materia']] = m['count']
        
        return jsonify(stats)

if __name__ == '__main__':
    init_db()
    print("Base de datos inicializada")
    app.run(debug=True, host='0.0.0.0', port=5000)