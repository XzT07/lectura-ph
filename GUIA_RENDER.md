# Publicar la demo en Render (enlace permanente y gratis)

Al terminar tendrás una dirección fija tipo `https://lectura-ph.onrender.com`
que funciona desde cualquier lugar, sin tu laptop encendida.

El plan tiene tres partes:
  A. Subir SOLO el modelo (300 MB) a un repositorio de modelos en Hugging Face
  B. Subir el CODIGO (liviano) a GitHub
  C. Conectar Render a ese GitHub para que lo publique

Suena a mucho, pero cada parte es corta. Vamos.

===================================================================
## PARTE A — Subir el modelo a Hugging Face (almacenamiento gratis)
===================================================================

Aunque Hugging Face ya no deja correr apps gratis, SI deja guardar modelos
gratis. Solo usamos su almacenamiento.

1. Crea cuenta en https://huggingface.co/join (si no la tienes)
2. Ve a https://huggingface.co/new (crear nuevo repositorio de MODELO)
   - Owner: tu usuario
   - Model name: `lectura-ph-modelo`
   - Visibility: **Public**
   - Pulsa **Create model**
3. En el repo creado, pestaña **Files** -> **Add file** -> **Upload files**
4. Sube tu `modelo_ph_antocianina.keras` (los 300 MB)
5. Abajo, **Commit changes to main**

Cuando termine, tu modelo vive en:
`https://huggingface.co/TU_USUARIO/lectura-ph-modelo`

**Anota el identificador** `TU_USUARIO/lectura-ph-modelo`, lo necesitaras en la
parte C.

===================================================================
## PARTE B — Subir el codigo a GitHub
===================================================================

Subiras estos archivos (todos livianos, el modelo NO va aqui):
  - app.py
  - requirements.txt
  - render.yaml

1. Crea cuenta en https://github.com/join (si no la tienes)
2. Ve a https://github.com/new
   - Repository name: `lectura-ph`
   - **Public**
   - Marca **Add a README file**
   - Pulsa **Create repository**
3. En el repo, boton **Add file** -> **Upload files**
4. Arrastra `app.py`, `requirements.txt` y `render.yaml`
5. Abajo, **Commit changes**

===================================================================
## PARTE C — Conectar Render
===================================================================

1. Crea cuenta en https://render.com (puedes entrar con tu GitHub, es lo mas facil)
2. En el panel, **New +** -> **Web Service**
3. Conecta tu cuenta de GitHub y elige el repo `lectura-ph`
4. Render detecta el `render.yaml` casi todo solo. Verifica:
   - **Runtime:** Python
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`
   - **Plan:** Free
5. Antes de crear, ve a **Environment** (variables de entorno) y agrega una:
   - Key: `REPO_MODELO`
   - Value: `TU_USUARIO/lectura-ph-modelo`  (el de la parte A)
6. Pulsa **Create Web Service**

Render empieza a construir. Tarda varios minutos (instala TensorFlow, descarga
tu modelo). Puedes ver el avance en la pestaña **Logs**.

Cuando termine, arriba aparece tu enlace:
`https://lectura-ph.onrender.com` (o similar)

Ese enlace:
- Funciona desde cualquier punto del pais o del mundo
- No necesita tu laptop
- Lo puedes convertir en QR para el stand

===================================================================
## Notas importantes
===================================================================

**El plan free "duerme" la app** tras 15 minutos sin uso. Al volver a abrirla,
tarda ~1 minuto en despertar y luego va normal. Para la feria: abre el enlace
unos minutos ANTES de presentar, para que este despierto.

**Primera carga lenta:** la primera vez que arranca descarga el modelo de 300 MB,
puede tardar. Las siguientes son mas rapidas.

**Memoria:** el plan free da 512 MB de RAM. TensorFlow + tu modelo estan al
limite. Si la app se cae por memoria (lo veras en Logs como "out of memory" o
"killed"), avisame y ajustamos: hay formas de reducir el consumo.

**El aviso de siempre:** el modelo se entreno con la escala de color de los
tubos. Para demostrar, usa los parches sinteticos o imagenes bien encuadradas.
