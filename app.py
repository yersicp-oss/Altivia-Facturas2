"""ALTIVIA Registro Eléctrico: Streamlit ligero, sin OCR ni IA."""
import base64
import binascii
import hashlib
import io
import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
import streamlit.components.v1 as components
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseUpload

ROOT_DEFAULT = "1rOoOcJ9SXHHc9BX6tLHjDIfE2_SqkeA1"
SCOPES = ["https://www.googleapis.com/auth/drive"]
MAX_BYTES = 12 * 1024 * 1024
EXTENSIONS = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".pdf": "application/pdf"}
CATEGORIES = ("BAJA TENSION", "MEDIA TENSION")
COMPONENT = components.declare_component("altivia_electrico_files_v1", path=str(Path(__file__).parent / "camera_native"))

st.set_page_config(page_title="ALTIVIA | Registro eléctrico", page_icon="📷", layout="wide")
st.markdown("""<style>
.block-container{max-width:1000px;padding-top:1.1rem}
.brand{text-align:center;margin:1rem 0 1.4rem}
.brand .a{font-size:clamp(2.3rem,5vw,3.5rem);font-weight:850;color:#102c4b;letter-spacing:.055em;line-height:1.14}
.brand .b{font-size:clamp(1.35rem,3vw,1.95rem);font-weight:550;color:#1767ab;margin-top:3px}
div.stButton>button[kind="primary"],div.stFormSubmitButton>button[kind="primary"]{background:#1767ab;border-color:#1767ab}
@media(max-width:590px){.brand .a{font-size:clamp(2.99rem,6.5vw,4.55rem)}.brand .b{font-size:clamp(1.755rem,3.9vw,2.535rem)}}
</style>""", unsafe_allow_html=True)


class DriveError(RuntimeError):
    """Error seguro que puede mostrarse al usuario."""


def now_peru():
    return datetime.now(ZoneInfo("America/Lima"))


def filename_base(description, category):
    """Nombre estable sin rutas ni caracteres ilegales."""
    cleaned = unicodedata.normalize("NFKC", description.strip())
    cleaned = re.sub(r"[\\/\x00-\x1f:*?\"<>|]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .-")
    if not cleaned:
        raise ValueError("Escribe una descripción válida (mínimo un carácter).")
    if len(cleaned) > 120:
        raise ValueError("La descripción debe tener como máximo 120 caracteres.")
    if category not in CATEGORIES:
        raise ValueError("Selecciona Baja tensión o Media tensión.")
    return f"{cleaned}-{category}"


def parse_upload(value):
    if not isinstance(value, dict) or not isinstance(value.get("data"), str):
        raise ValueError("El archivo recibido no tiene un formato válido.")
    try:
        raw = base64.b64decode(value["data"], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("No se pudo decodificar el archivo.") from exc
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("El archivo está vacío o supera los 12 MB.")
    filename = str(value.get("filename", ""))
    ext = Path(filename).suffix.lower()
    if ext not in EXTENSIONS:
        raise ValueError("Adjunta un archivo JPG, PNG, WEBP o PDF.")
    # Validación básica de contenido, no confiar solo en la extensión.
    if ext == ".pdf" and not raw.startswith(b"%PDF"):
        raise ValueError("El documento seleccionado no es un PDF válido.")
    if ext in (".jpg", ".jpeg") and not raw.startswith(b"\xff\xd8"):
        raise ValueError("La foto no tiene formato JPEG válido.")
    if ext == ".png" and not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("La imagen no tiene formato PNG válido.")
    if ext == ".webp" and not (raw.startswith(b"RIFF") and raw[8:12] == b"WEBP"):
        raise ValueError("La imagen no tiene formato WEBP válido.")
    return raw, ext, filename


def drive_service():
    required = ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN")
    missing = [key for key in required if not st.secrets.get(key)]
    if missing:
        raise DriveError("Faltan credenciales en Streamlit Secrets: " + ", ".join(missing))
    try:
        creds = Credentials(
            token=None,
            refresh_token=st.secrets["GOOGLE_REFRESH_TOKEN"],
            token_uri="https://oauth2.googleapis.com/token",
            client_id=st.secrets["GOOGLE_CLIENT_ID"],
            client_secret=st.secrets["GOOGLE_CLIENT_SECRET"],
            scopes=SCOPES,
        )
        creds.refresh(Request())
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception as exc:
        logging.exception("Google OAuth failed")
        raise DriveError("Falló la autorización de Google. Verifica el Refresh Token y los permisos OAuth.") from exc


def drive_folder(service, name, parent):
    safe_name = name.replace("\\", "\\\\").replace("'", "\\'")
    q = (f"name = '{safe_name}' and mimeType = 'application/vnd.google-apps.folder' "
         f"and '{parent}' in parents and trashed = false")
    res = service.files().list(q=q, spaces="drive", fields="nextPageToken, files(id,name)",
                               pageSize=100, supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
    if res.get("files"):
        return res["files"][0]["id"]
    folder = service.files().create(body={"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent]},
                                    fields="id", supportsAllDrives=True).execute()
    return folder["id"]


def next_filename(service, parent, base, extension):
    """Asigna -1, -2, -3... contando las coincidencias ya presentes en la carpeta."""
    q = f"'{parent}' in parents and trashed = false"
    token = None
    numbers = set()
    pattern = re.compile(rf"^{re.escape(base)}-(\d+)\.[A-Za-z0-9]+$", re.IGNORECASE)
    while True:
        res = service.files().list(q=q, fields="nextPageToken,files(name)", pageSize=1000,
                                   pageToken=token, supportsAllDrives=True,
                                   includeItemsFromAllDrives=True).execute()
        for item in res.get("files", []):
            match = pattern.fullmatch(item.get("name", ""))
            if match:
                numbers.add(int(match.group(1)))
        token = res.get("nextPageToken")
        if not token:
            break
    num = 1
    while num in numbers:
        num += 1
    return f"{base}-{num}{extension}"


def save_drive(raw, extension, description, category, when=None):
    base = filename_base(description, category)
    when = when or now_peru()
    root = st.secrets.get("GOOGLE_DRIVE_FOLDER_ID", ROOT_DEFAULT)
    try:
        service = drive_service()
        # Comprueba que la cuenta OAuth, no solo el navegador, pueda acceder al directorio.
        service.files().get(fileId=root, fields="id,name,mimeType", supportsAllDrives=True).execute()
        month = drive_folder(service, when.strftime("%Y-%m"), root)
        day = drive_folder(service, when.strftime("%d-%m-%Y"), month)
        filename = next_filename(service, day, base, extension)
        upload = MediaIoBaseUpload(io.BytesIO(raw), mimetype=EXTENSIONS[extension], resumable=False)
        file = service.files().create(body={"name": filename, "parents": [day]}, media_body=upload,
                                      fields="id,webViewLink,name", supportsAllDrives=True).execute()
        file_id = file["id"]
        return {"name": filename, "id": file_id,
                "url": file.get("webViewLink") or f"https://drive.google.com/file/d/{file_id}/view",
                "month": when.strftime("%Y-%m"), "day": when.strftime("%d-%m-%Y")}
    except DriveError:
        raise
    except HttpError as exc:
        logging.exception("Drive upload HTTP error")
        code = getattr(exc.resp, "status", "desconocido")
        raise DriveError(f"Google Drive rechazó el guardado (HTTP {code}). Verifica permisos y cuota.") from exc
    except Exception as exc:
        logging.exception("Drive upload unexpected error")
        raise DriveError("No se pudo guardar el documento. Revisa los registros privados de Streamlit.") from exc


def admin():
    st.subheader("Ingreso administrativo")
    if st.button("← Volver"):
        st.session_state.page = "main"
        st.rerun()
    expected = st.secrets.get("ADMIN_PASSWORD", "")
    if not expected:
        st.info("Para activar el diagnóstico configura ADMIN_PASSWORD en Streamlit Secrets.")
        return
    if not st.session_state.get("admin_ok"):
        with st.form("login"):
            password = st.text_input("Contraseña", type="password")
            go = st.form_submit_button("Ingresar")
        if go:
            import hmac
            if hmac.compare_digest(password, expected):
                st.session_state.admin_ok = True
                st.rerun()
            else:
                st.error("Contraseña incorrecta")
        return
    if st.button("Cerrar sesión"):
        st.session_state.admin_ok = False
        st.rerun()
    st.markdown("#### Diagnóstico de Google Drive")
    if st.button("Comprobar acceso a carpeta"):
        try:
            service = drive_service()
            root = st.secrets.get("GOOGLE_DRIVE_FOLDER_ID", ROOT_DEFAULT)
            info = service.files().get(fileId=root, fields="id,name,mimeType,capabilities(canAddChildren)",
                                       supportsAllDrives=True).execute()
            st.success(f"Carpeta accesible: {info.get('name', 'sin nombre')}")
            st.write("Permiso para agregar archivos:", info.get("capabilities", {}).get("canAddChildren", "no determinado"))
        except Exception as exc:
            if isinstance(exc, DriveError):
                st.error(str(exc))
            else:
                st.error("Falló la consulta a Drive; revisa los logs privados.")


def main():
    st.markdown('<div class="brand"><div class="a">ALTIVIA</div><div class="b">Registro eléctrico</div></div>', unsafe_allow_html=True)
    item = COMPONENT(key="filepicker", default=None, height=350)
    if not item:
        return
    signature = hashlib.sha256((str(item.get("data", "")) + str(item.get("filename", ""))).encode("utf-8")).hexdigest()
    if st.session_state.get("last_upload_signature") != signature:
        st.session_state.last_upload_signature = signature
        st.session_state.current_item = item
        st.session_state.saved_signature = None
    try:
        raw, extension, filename = parse_upload(st.session_state.current_item)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.divider()
    st.success(f"Archivo seleccionado: {filename}")
    # Control independiente del formulario: evita desincronización entre iframe y submit.
    # Solo se presentan valores válidos, sin opción provisional ni valor vacío.
    tension = st.selectbox(
        "Tipo de instalación *",
        options=CATEGORIES,
        format_func=lambda x: "Baja tensión" if x == "BAJA TENSION" else "Media tensión",
        key="electric_voltage_category",
    )
    with st.form("metadata"):
        description = st.text_input("Descripción del documento *", max_chars=120,
                                    placeholder="Ejemplo: Poste N° 01", help="Mínimo 1 carácter")
        submit = st.form_submit_button("Guardar en Google Drive", type="primary", use_container_width=True)
    if submit:
        if not description.strip():
            st.warning("Ingresa una descripción de al menos un carácter.")
            return
        if st.session_state.get("saved_signature") == signature:
            st.info("Este archivo ya fue guardado en esta sesión. Toma otra foto para continuar.")
            return
        category = tension
        with st.spinner("Guardando documento en Google Drive..."):
            try:
                result = save_drive(raw, extension, description, category)
                st.session_state.saved_signature = signature
                st.success(f"Documento guardado: {result['name']}")
                st.caption(f"Carpeta: {result['month']} / {result['day']}")
                st.link_button("Ver respaldo en Google Drive", result["url"])
            except (DriveError, ValueError) as exc:
                st.error(str(exc))


_, right = st.columns([4, 1])
if right.button("Ingreso administrativo", use_container_width=True):
    st.session_state.page = "admin"
    st.rerun()
if st.session_state.get("page") == "admin":
    admin()
else:
    main()
