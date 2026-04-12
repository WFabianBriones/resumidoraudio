#!/bin/bash

echo "🚀 EduTranscribe - Configuración Optimizada"
echo "=========================================="
echo ""

# Verificar modelo Llama 3.1-8B
if ! ollama list | grep -q "llama3.1:8b"; then
    echo "📥 Descargando Llama 3.1-8B (~4.7GB)..."
    ollama pull llama3.1:8b
    echo "✓ Modelo descargado"
fi

# Limpiar procesos anteriores
pkill -f "python3 app.py" 2>/dev/null
sleep 2

# Verificar Ollama
if ! pgrep -x "ollama" > /dev/null; then
    echo "🔄 Iniciando Ollama..."
    ollama serve > /dev/null 2>&1 &
    sleep 5
fi

echo ""
echo "⚙️  Configuración Actual:"
echo "  • Whisper Medium: CPU (transcripción ~15-20min/hora)"
echo "  • Llama 3.1-8B: GPU (resumen mejorado ~8-12min)"
echo "  • Validación anti-alucinación: ACTIVADA"
echo "  • Post-procesamiento: GENÉRICO"
echo ""
echo "🎯 Calidad esperada:"
echo "  • Transcripción: 88% precisión (Medium)"
echo "  • Resumen: 88% precisión (Llama 3.1-8B)"
echo "  • Detección de alucinaciones: ACTIVA"
echo ""
echo "⏱️  Tiempo estimado (clase de 60 min):"
echo "  • Transcripción: ~15-20 minutos"
echo "  • Resumen: ~8-12 minutos"
echo "  • Total: ~25-30 minutos"
echo ""

cd ~/grabador-clases
source venv/bin/activate
python3 app.py
