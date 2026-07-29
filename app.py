"""
Sistema de lectura óptica de pH en heridas — demo con modelo real.

Interfaz de instrumento para la Feria de Proyectos de Ingeniería 2026.
Analiza la foto de un parche colorimétrico de quitosano con antocianina y
emite un diagnóstico preliminar del pH con su nivel de confianza.

REQUISITOS:
    py -3.12 -m pip install tensorflow gradio pillow numpy opencv-python

USO:
    py -3.12 demo_local.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import gradio as gr
import cv2
# Limitar hilos de OpenCV: en un servidor con poca RAM, menos hilos = menos
# memoria y suficiente para esta tarea ligera.
cv2.setNumThreads(1)
# Runtime ligero de TFLite (mucho menos memoria que TensorFlow completo).
# En Render usamos el paquete "ai-edge-litert" que provee el interprete.
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    # alternativa: interprete de tensorflow si estuviera disponible
    from tensorflow.lite import Interpreter

MODELO = "modelo_ph_antocianina.tflite"
IMG_SIZE = 227
PH_VALORES = [4, 5, 6, 7, 8, 9, 10]

COLOR_PH = {
    4:  (234, 103, 156), 5: (223, 152, 194), 6: (174, 196, 219),
    7:  (142, 181, 210), 8: (84, 164, 162),  9: (80, 95, 132),
    10: (21, 35, 88),
}

# Diagnóstico preliminar por nivel de pH y color de estado (verde/ámbar/rojo)
DIAGNOSTICO = {
    4:  ("Ambiente muy ácido",
         "Por debajo del rango cutáneo habitual. Puede indicar exceso de acidez "
         "local; correlacionar con el estado del tejido.", "#2aa88b"),
    5:  ("Ambiente ácido",
         "Compatible con piel sana y entorno favorable a la cicatrización. "
         "Continuar el seguimiento habitual.", "#2aa88b"),
    6:  ("Ambiente ligeramente ácido",
         "Zona de transición. Cicatrización posible; vigilar la evolución en las "
         "siguientes revisiones.", "#e0a92e"),
    7:  ("Ambiente neutro",
         "Punto de inflexión. Vigilar de cerca: la neutralización sostenida "
         "puede preceder a un viraje alcalino.", "#e0a92e"),
    8:  ("Ambiente ligeramente alcalino",
         "Signo de alerta temprana. La alcalinización se asocia a retraso en la "
         "cicatrización; valorar posible colonización.", "#e06b7a"),
    9:  ("Ambiente alcalino",
         "Compatible con herida crónica o infección. Se sugiere evaluación "
         "clínica y considerar cultivo.", "#e06b7a"),
    10: ("Ambiente muy alcalino",
         "Fuertemente asociado a infección o cronicidad. Evaluación clínica "
         "prioritaria.", "#e06b7a"),
}


def recortar_parche(img_rgb):
    """Aisla la zona coloreada del parche, descartando el fondo del pozo."""
    img = img_rgb.astype("float32")
    h, w, _ = img.shape
    mx = img.max(axis=2); mn = img.min(axis=2)
    sat = mx - mn
    brillo = img.mean(axis=2)
    score = sat.copy()
    score[brillo > 232] = 0
    score[brillo < 35] = 0
    score = cv2.GaussianBlur(score, (31, 31), 0)
    if score.max() < 3:
        m = min(h, w); cy, cx = h // 2, w // 2; r = m // 3
        return img_rgb[cy-r:cy+r, cx-r:cx+r]
    thr = np.percentile(score[score > 0], 90)
    ys, xs = np.where(score >= thr)
    cy, cx = int(ys.mean()), int(xs.mean())
    r = int(min(h, w) * 0.18)
    y0, y1 = max(0, cy-r), min(h, cy+r)
    x0, x1 = max(0, cx-r), min(w, cx+r)
    return img_rgb[y0:y1, x0:x1]


# El modelo TFLite se descarga desde el repositorio de Hugging Face al arrancar.
from huggingface_hub import hf_hub_download

REPO_MODELO = os.environ.get("REPO_MODELO", "XzT07/lectura-ph-modelo")

print("Descargando modelo TFLite desde Hugging Face...")
ruta_modelo = hf_hub_download(repo_id=REPO_MODELO, filename=MODELO)
print("Cargando modelo...")
interpreter = Interpreter(model_path=ruta_modelo)
interpreter.allocate_tensors()
_in = interpreter.get_input_details()[0]
_out = interpreter.get_output_details()[0]
print("Modelo cargado. Iniciando interfaz...")


def _predecir(x):
    """Inferencia con TFLite. x: array (1,227,227,3) float32."""
    interpreter.set_tensor(_in["index"], x.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(_out["index"])[0]


def analizar(imagen):
    if imagen is None:
        vacio = ("<div class='rx-empty'>Sin imagen cargada.<br>"
                 "Suba una fotografía del parche para obtener el diagnóstico "
                 "preliminar.</div>")
        return vacio, None

    import gc
    # PROTECCIÓN DE MEMORIA (crítica para el plan gratuito):
    # Las fotos de celular pueden ser de 50 MP. Se reduce la imagen a un tamaño
    # de trabajo pequeño LO ANTES POSIBLE, y se usa PIL (que decodifica de forma
    # más liviana que cargar todo a numpy). El color no se altera.
    imagen = imagen.convert("RGB")

    # Reducir de forma escalonada: si es enorme, usar thumbnail (in-place,
    # eficiente en memoria) para bajar el lado mayor a 800 px como máximo.
    MAX_LADO = 800
    if max(imagen.size) > MAX_LADO:
        imagen.thumbnail((MAX_LADO, MAX_LADO))   # in-place, libera el original

    arr = np.array(imagen)
    del imagen
    recorte = recortar_parche(arr)
    img = cv2.resize(recorte, (IMG_SIZE, IMG_SIZE))
    x = img.astype("float32")[None, ...]
    probs = _predecir(x)
    idx = int(np.argmax(probs))
    ph = PH_VALORES[idx]
    conf = float(probs[idx]) * 100
    titulo, texto, col = DIAGNOSTICO[ph]
    r, g, b = COLOR_PH[ph]

    # nivel de confianza cualitativo
    if conf >= 80:
        conf_txt = "alta"
    elif conf >= 55:
        conf_txt = "moderada"
    else:
        conf_txt = "baja — repetir la captura"

    barras = ""
    for i, pv in enumerate(PH_VALORES):
        cr, cg, cb = COLOR_PH[pv]
        pct = probs[i] * 100
        barras += (
            f"<div class='rx-bar'>"
            f"<span class='rx-bar-label'>pH {pv}</span>"
            f"<span class='rx-bar-track'><span class='rx-bar-fill' "
            f"style='width:{pct:.1f}%;background:rgb({cr},{cg},{cb})'></span></span>"
            f"<span class='rx-bar-pct'>{pct:.0f}%</span>"
            f"</div>"
        )

    html = f"""
    <div class='rx-report'>
      <div class='rx-report-head'>Diagnóstico preliminar</div>
      <div class='rx-ph-row'>
        <div class='rx-ph' style='color:{col}'>pH {ph}</div>
        <div class='rx-swatch' style='background:rgb({r},{g},{b})'></div>
      </div>
      <div class='rx-diag-title' style='color:{col}'>{titulo}</div>
      <div class='rx-diag-text'>{texto}</div>
      <div class='rx-conf'>
        Confianza del modelo: <b style='color:{col}'>{conf:.0f}%</b>
        <span class='rx-conf-q'>({conf_txt})</span>
      </div>
      <div class='rx-bars-title'>Probabilidad por nivel</div>
      <div class='rx-bars'>{barras}</div>
      <div class='rx-disclaimer'>Resultado orientativo. No sustituye la
      valoración de un profesional de salud.</div>
    </div>
    """
    recorte_vis = cv2.resize(recorte, (170, 170))
    # Liberar arrays grandes y forzar recolección de basura para no acumular
    # memoria entre peticiones (importante en el plan gratuito).
    del arr, img, x, recorte
    gc.collect()
    return html, recorte_vis


CSS = """
.gradio-container {
  background:
    radial-gradient(900px 560px at -8% -12%, rgba(34,161,137,.28), transparent 55%),
    radial-gradient(680px 480px at 108% -6%, rgba(24,90,120,.22), transparent 60%),
    linear-gradient(165deg,#0d2748 0%,#0a2040 34%,#081733 66%,#050f22 100%)
    !important;
  min-height: 100vh;
  font-family: 'Inter', system-ui, sans-serif !important;
}
.rx-header { padding: 8px 4px 0; }
.rx-eyebrow { font-size:11px; font-weight:700; letter-spacing:.14em;
  text-transform:uppercase; color:#ffd35c; }
.rx-title { font-size:clamp(23px,3.4vw,34px); font-weight:800; line-height:1.05;
  letter-spacing:-.015em; margin:6px 0 2px; color:#fff; }
.rx-sub { color:#c7d6ef; font-size:14px; margin-top:8px; max-width:64ch; }

/* instrucciones de uso */
.rx-steps { display:flex; gap:12px; margin:18px 4px 4px; flex-wrap:wrap; }
.rx-step { flex:1; min-width:200px; background:rgba(18,42,94,.5);
  border:1px solid rgba(255,255,255,.09); border-left:3px solid #f5b400;
  border-radius:12px; padding:13px 15px; }
.rx-step-n { font-family:'JetBrains Mono',monospace; font-size:11px;
  font-weight:700; color:#ffd35c; letter-spacing:.05em; }
.rx-step-t { color:#e8eefb; font-size:13.5px; font-weight:600; margin-top:4px; }
.rx-step-d { color:#9fb2d8; font-size:12.5px; margin-top:3px; line-height:1.5; }

/* escala */
.rx-scale-wrap { margin: 18px 4px 4px; }
.rx-scale-label { font-size:11px; text-transform:uppercase; letter-spacing:.09em;
  color:#c7d6ef; font-weight:700; margin-bottom:8px; }
.rx-scale { display:flex; border-radius:12px; overflow:hidden; height:44px;
  box-shadow:0 10px 28px rgba(0,0,0,.4), 0 0 0 1px rgba(255,255,255,.06); }
.rx-scale div { flex:1; display:flex; align-items:flex-end; justify-content:center;
  padding-bottom:5px; font-size:11px; font-weight:700;
  font-family:'JetBrains Mono',monospace; color:rgba(255,255,255,.92);
  text-shadow:0 1px 4px rgba(0,0,0,.5); }

.rx-card { background:linear-gradient(180deg,#122a5e,rgba(18,42,94,.55)) !important;
  border:1px solid rgba(255,255,255,.10) !important; border-radius:16px !important; }

/* reporte */
.rx-empty { color:#7f93bd; text-align:center; padding:46px 22px; font-size:14px;
  line-height:1.6; }
.rx-report { padding:4px 6px; }
.rx-report-head { font-size:11px; text-transform:uppercase; letter-spacing:.11em;
  color:#9fb2d8; font-weight:700; margin-bottom:12px; }
.rx-ph-row { display:flex; align-items:center; justify-content:space-between; }
.rx-ph { font-family:'JetBrains Mono',monospace; font-size:48px; font-weight:700;
  line-height:1; }
.rx-swatch { width:52px; height:52px; border-radius:12px;
  box-shadow:0 4px 14px rgba(0,0,0,.35); }
.rx-diag-title { font-size:18px; font-weight:700; margin-top:14px; }
.rx-diag-text { color:#c7d6ef; font-size:13.5px; line-height:1.6; margin-top:6px; }
.rx-conf { color:#c7d6ef; font-size:13px; margin-top:16px;
  padding-top:14px; border-top:1px solid rgba(255,255,255,.09); }
.rx-conf-q { color:#9fb2d8; }
.rx-bars-title { font-size:11px; text-transform:uppercase; letter-spacing:.08em;
  color:#9fb2d8; font-weight:700; margin:16px 0 10px; }
.rx-bar { display:grid; grid-template-columns:44px 1fr 40px; align-items:center;
  gap:10px; margin-bottom:7px; font-family:'JetBrains Mono',monospace;
  font-size:12px; color:#c7d6ef; }
.rx-bar-track { height:7px; background:rgba(5,11,28,.5); border-radius:100px;
  overflow:hidden; }
.rx-bar-fill { display:block; height:100%; border-radius:100px; transition:width .5s; }
.rx-bar-pct { text-align:right; }
.rx-disclaimer { color:#7f93bd; font-size:11px; font-style:italic; margin-top:16px;
  padding-top:12px; border-top:1px solid rgba(255,255,255,.07); line-height:1.5; }

.rx-foot { color:#7f93bd; font-size:11.5px; text-align:center; margin-top:18px;
  padding-top:13px; border-top:1px solid rgba(255,255,255,.08); line-height:1.7; }
.rx-foot b { color:#c7d6ef; }
"""


def _scale_html():
    celdas = ""
    for ph in PH_VALORES:
        r, g, b = COLOR_PH[ph]
        celdas += f"<div style='background:rgb({r},{g},{b})'>{ph}</div>"
    return f"<div class='rx-scale'>{celdas}</div>"


with gr.Blocks(css=CSS, title="Lectura óptica de pH — Feria de Ingeniería 2026",
               theme=gr.themes.Base()) as demo:
    # Fuerza la cámara TRASERA en móviles: intercepta la petición de cámara del
    # navegador y pide facingMode 'environment' (trasera). Sin esto, muchos
    # celulares abren la frontal por defecto en la webcam de Gradio.
    gr.HTML("""
    <script>
    (function(){
      if (!navigator.mediaDevices) return;
      const orig = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
      navigator.mediaDevices.getUserMedia = function(constraints){
        try {
          if (constraints && constraints.video) {
            if (constraints.video === true) constraints.video = {};
            // preferir la cámara trasera
            constraints.video.facingMode = { ideal: "environment" };
          }
        } catch(e){}
        return orig(constraints);
      };
    })();
    </script>
    """)
    gr.HTML(f"""
      <div class='rx-header'>
        <div class='rx-eyebrow'>Feria de Proyectos de Ingeniería 2026 · U. Latina de Panamá</div>
        <div class='rx-title'>Lectura óptica de pH en heridas</div>
        <div class='rx-sub'>Sistema de apoyo diagnóstico basado en un parche
        colorimétrico de quitosano con antocianina. La red neuronal estima el pH
        del lecho de la herida a partir del color del parche y emite un
        diagnóstico preliminar.</div>
      </div>

      <div class='rx-steps'>
        <div class='rx-step'>
          <div class='rx-step-n'>PASO 1</div>
          <div class='rx-step-t'>Fotografíe el parche</div>
          <div class='rx-step-d'>Con buena luz y el parche llenando el encuadre,
          sin retirar el vendaje.</div>
        </div>
        <div class='rx-step'>
          <div class='rx-step-n'>PASO 2</div>
          <div class='rx-step-t'>Cargue la imagen</div>
          <div class='rx-step-d'>Arrastre o seleccione la foto en el panel de la
          izquierda.</div>
        </div>
        <div class='rx-step'>
          <div class='rx-step-n'>PASO 3</div>
          <div class='rx-step-t'>Lea el diagnóstico</div>
          <div class='rx-step-d'>El sistema estima el pH y su interpretación
          clínica con el nivel de confianza.</div>
        </div>
      </div>

      <div class='rx-scale-wrap'>
        <div class='rx-scale-label'>Escala de referencia · respuesta de la antocianina · pH 4-10</div>
        {_scale_html()}
      </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            # sources: subir archivo (usa la cámara nativa del móvil, donde sí
            # se elige la trasera) y webcam. En móvil, "upload" abre la cámara
            # del teléfono con control real de cuál cámara usar.
            entrada = gr.Image(type="pil", label="Fotografía del parche",
                               sources=["upload", "webcam"],
                               elem_classes="rx-card", height=330)
            recorte_out = gr.Image(label="Región analizada por el modelo",
                                   height=170, elem_classes="rx-card")
        with gr.Column(scale=1):
            salida = gr.HTML("<div class='rx-empty'>Sin imagen cargada.<br>"
                             "Suba una fotografía del parche para obtener el "
                             "diagnóstico preliminar.</div>", elem_classes="rx-card")

    gr.HTML("""
      <div class='rx-foot'>
        Escuela de Ingeniería Biomédica · Universidad Latina de Panamá ·
        El modelo se ejecuta localmente sobre la red neuronal entrenada.
        Resultado orientativo, no sustituye la valoración médica.
      </div>
    """)

    entrada.change(fn=analizar, inputs=entrada, outputs=[salida, recorte_out],
                   concurrency_limit=1)

if __name__ == "__main__":
    # Render asigna el puerto por la variable de entorno PORT.
    # server_name 0.0.0.0 hace que escuche en toda la red del contenedor.
    # max_threads y una cola con límite evitan que varias imágenes se procesen
    # a la vez y disparen la memoria en el plan gratuito.
    import os
    puerto = int(os.environ.get("PORT", 7860))
    demo.queue(default_concurrency_limit=1, max_size=8)
    demo.launch(server_name="0.0.0.0", server_port=puerto, max_threads=2)
