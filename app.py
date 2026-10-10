"""ALTIVIA Facturas Streamlit Lite — versión prototipo sin almacenamiento persistente."""
import hashlib
import base64
import io
import json
import re
import sqlite3
from datetime import date, datetime
from pathlib import Path
from datetime import timezone
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from googleapiclient.errors import HttpError

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types
from PIL import Image, ImageOps, UnidentifiedImageError

st.set_page_config(page_title="ALTIVIA | Facturas", page_icon="🧾", layout="wide")
DB_PATH = Path("altivia_demo.db")
DRIVE_ROOT_ID = "1rOoOcJ9SXHHc9BX6tLHjDIfE2_SqkeA1"
TYPES = ["Factura", "Boleta", "Nota de crédito", "Otro"]
SCHEMA = {"tipo": "", "serie": "", "numero": "", "fecha": "", "proveedor": "", "ruc": "", "total": ""}
CAMARA_NATIVA = components.declare_component("altivia_native_camera_v6", path=str(Path(__file__).parent / "camera_native"))


st.markdown("""<style>
.block-container {max-width:1150px; padding-top:2rem}
h1, h2, h3 {color:#102c4b}
div.stButton > button[kind="primary"] {background:#1767ab; border-color:#1767ab}
</style>""", unsafe_allow_html=True)


def db():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("CREATE TABLE IF NOT EXISTS members (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, active INTEGER NOT NULL DEFAULT 1)")
    conn.execute("""CREATE TABLE IF NOT EXISTS invoices (
        id INTEGER PRIMARY KEY, member_id INTEGER NOT NULL, tipo TEXT NOT NULL,
        serie TEXT NOT NULL, numero TEXT NOT NULL, fecha TEXT NOT NULL,
        proveedor TEXT NOT NULL, ruc TEXT NOT NULL, total REAL NOT NULL,
        created_at TEXT NOT NULL, UNIQUE(tipo,ruc,serie,numero),
        FOREIGN KEY(member_id) REFERENCES members(id))""")
    columns = {r[1] for r in conn.execute("PRAGMA table_info(invoices)").fetchall()}
    for column in ("drive_file_id", "drive_url", "drive_status"):
        if column not in columns:
            conn.execute(f"ALTER TABLE invoices ADD COLUMN {column} TEXT")
    conn.commit()
    return conn


def list_members():
    with db() as conn:
        return conn.execute("SELECT id,name FROM members WHERE active=1 ORDER BY name").fetchall()


def prep_image(data: bytes, max_side=1400):
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((max_side, max_side))
        output = io.BytesIO()
        im.save(output, format="JPEG", quality=76, optimize=True)
    return output.getvalue()


def pdf_images(data: bytes):
    # pypdfium2 is installed by Streamlit cloud only when reading PDFs
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise ValueError("Para leer PDF agrega pypdfium2 a requirements.txt")
    doc = pdfium.PdfDocument(data)
    try:
        if len(doc) == 0:
            raise ValueError("PDF vacío")
        imgs = []
        for idx in range(min(2, len(doc))):
            page = doc[idx]
            pil = page.render(scale=1.5).to_pil()
            buf = io.BytesIO()
            pil.convert("RGB").save(buf, "JPEG", quality=83)
            imgs.append(prep_image(buf.getvalue()))
            page.close()
        return imgs
    finally:
        doc.close()


def parse_json(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    obj = json.loads(raw)
    return {k: str(obj.get(k) or "").strip() for k in SCHEMA}


def recognize(upload):
    """Extrae exclusivamente los siete campos usando Gemini, sin OCR local."""
    key = st.secrets.get("GEMINI_API_KEY", "")
    if not key or str(key).startswith("TU_CLAVE"):
        raise ValueError("Falta configurar GEMINI_API_KEY en Secrets de Streamlit")
    content = upload.getvalue()
    if len(content) > 12 * 1024 * 1024:
        raise ValueError("Máximo 12 MB por archivo")
    if content.startswith(b"%PDF"):
        pictures = pdf_images(content)
    else:
        try:
            pictures = [prep_image(content)]
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("La imagen está dañada o no es compatible") from exc

    instructions = (
        "Lee el comprobante peruano de la(s) imagen(es). Responde SOLO JSON válido "
        "con estas siete claves exactas: tipo, serie, numero, fecha, proveedor, ruc, total. "
        "tipo debe ser Factura, Boleta, Nota de crédito u Otro. "
        "fecha en AAAA-MM-DD; total numérico sin símbolos, con punto decimal. "
        "Separa la serie del número correlativo. Conserva ceros a la izquierda. "
        "RUC de exactamente 11 dígitos si aparece. No inventes información, "
        "si falta o es ilegible escribe cadena vacía. "
        "Si hay varios importes identifica únicamente el importe total a pagar."
    )
    contents = [instructions] + [
        types.Part.from_bytes(data=image, mime_type="image/jpeg") for image in pictures
    ]
    model = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
    try:
        with genai.Client(api_key=key, http_options=types.HttpOptions(timeout=30000)) as client:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=480,
                    temperature=0,
                ),
            )
        if not response.text:
            raise ValueError("Gemini no devolvió texto. Prueba otra fotografía.")
        return parse_json(response.text)
    except ValueError:
        raise
    except Exception as exc:
        # No mostrar claves, payloads o trazas del proveedor en la interfaz.
        msg = str(exc).lower()
        if "429" in msg or "resource_exhausted" in msg:
            raise ValueError("Gemini alcanzó el límite de solicitudes o cuota. Intenta más tarde.") from None
        if "403" in msg or "permission_denied" in msg or "api_key_invalid" in msg or "400 api key" in msg:
            raise ValueError("La clave de Gemini no tiene acceso. Revisa Google AI Studio y Secrets.") from None
        if "404" in msg or "not found" in msg:
            raise ValueError("Modelo de Gemini no disponible. Revisa GEMINI_MODEL.") from None
        raise ValueError("No se pudo completar la lectura con Gemini. Revisa los logs privados de Streamlit.") from None


def money(v):
    try:
        n = float(str(v).replace(",", "."))
    except ValueError as exc:
        raise ValueError("El importe debe ser numérico") from exc
    if not 0 < n < 1_000_000_000:
        raise ValueError("Importe total fuera de rango")
    return n


class DriveBackupError(Exception):
    """Safe-to-display diagnostics. Never include credential or raw API payload."""


def drive_error(phase, exc):
    if isinstance(exc, RefreshError):
        raise DriveBackupError(
            f"{phase}: no fue posible renovar la autorización OAuth. "
            "Verifica que CLIENT_ID, CLIENT_SECRET y REFRESH_TOKEN sean del mismo cliente, "
            "y que el token siga vigente."
        ) from exc
    if isinstance(exc, HttpError):
        status = exc.resp.status
        if status == 403:
            hint = "Sin permiso API para la carpeta o alcance OAuth insuficiente."
        elif status == 404:
            hint = "La cuenta OAuth no puede ver la carpeta indicada (o su ID no existe)."
        elif status == 401:
            hint = "La autorización de Google no es válida."
        elif status == 429:
            hint = "Google Drive está limitando temporalmente las solicitudes."
        else:
            hint = "Revisa los permisos, la cuota y la configuración de Google Drive."
        raise DriveBackupError(f"{phase}: Google Drive respondió HTTP {status}. {hint}") from exc
    raise DriveBackupError(f"{phase}: fallo de conexión o configuración ({type(exc).__name__}).") from exc


def drive_client():
    """Utiliza OAuth de usuario: el navegador y la API pueden tener accesos distintos."""
    settings = st.secrets
    required = ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN")
    missing = [key for key in required if not settings.get(key)]
    if missing:
        raise DriveBackupError("Faltan secretos en Streamlit: " + ", ".join(missing))
    # Este scope exige que el refresh token se haya autorizado con Drive completo.
    # No se puede ampliar el permiso de un refresh token ya emitido.
    creds = Credentials(
        token=None,
        refresh_token=settings["GOOGLE_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=settings["GOOGLE_CLIENT_ID"],
        client_secret=settings["GOOGLE_CLIENT_SECRET"],
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    try:
        creds.refresh(Request())
    except Exception as exc:
        drive_error("Renovación OAuth", exc)
    try:
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception as exc:
        drive_error("Conexión API", exc)


def drive_root_id():
    return str(st.secrets.get("GOOGLE_DRIVE_FOLDER_ID", DRIVE_ROOT_ID)).strip()


def drive_check_root(service):
    """Comprueba acceso de la cuenta OAuth, no del navegador abierto."""
    try:
        folder = service.files().get(
            fileId=drive_root_id(),
            fields="id,name,mimeType,capabilities(canAddChildren)",
            supportsAllDrives=True,
        ).execute()
    except Exception as exc:
        drive_error("Acceso a carpeta raíz", exc)
    if folder.get("mimeType") != "application/vnd.google-apps.folder":
        raise DriveBackupError("La carpeta configurada no es una carpeta de Google Drive.")
    if folder.get("capabilities", {}).get("canAddChildren") is False:
        raise DriveBackupError("La cuenta OAuth puede ver la carpeta pero no crear archivos dentro.")
    return folder


def drive_folder(service, name, parent):
    escaped = name.replace("\\", "\\\\").replace("'", "\\'")
    q = (f"name = '{escaped}' and mimeType = "
         f"'application/vnd.google-apps.folder' and '{parent}' in parents and trashed = false")
    try:
        result = service.files().list(
            q=q, fields="nextPageToken,files(id,name)", pageSize=100,
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        if result.get("files"):
            return result["files"][0]["id"]
        created = service.files().create(
            body={"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent]},
            fields="id", supportsAllDrives=True,
        ).execute()
        return created["id"]
    except Exception as exc:
        drive_error("Crear/buscar carpeta " + name, exc)


def drive_upload(file_bytes, original_name, invoice_values):
    """Respalda el original en FACTURAS / AAAA-MM / DD-MM-AAAA."""
    service = drive_client()
    drive_check_root(service)
    issued = date.fromisoformat(invoice_values["fecha"])
    month = drive_folder(service, issued.strftime("%Y-%m"), drive_root_id())
    day = drive_folder(service, issued.strftime("%d-%m-%Y"), month)
    ext = Path(original_name).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp", ".pdf"):
        raise ValueError("Extensión de archivo no admitida")
    name = f'{invoice_values["serie"].upper()}-{invoice_values["numero"].upper()}-{invoice_values["ruc"]}{ext}'
    mime = {".jpg":"image/jpeg", ".jpeg":"image/jpeg", ".png":"image/png", ".webp":"image/webp", ".pdf":"application/pdf"}[ext]
    escaped = name.replace("\\", "\\\\").replace("'", "\\'")
    q = f"name = '{escaped}' and '{day}' in parents and trashed = false"
    try:
        existing = service.files().list(
            q=q, fields="nextPageToken,files(id,webViewLink)", pageSize=10,
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute().get("files", [])
        if existing:
            fid = existing[0]["id"]
            return fid, existing[0].get("webViewLink") or f"https://drive.google.com/file/d/{fid}/view"
        media = MediaIoBaseUpload(io.BytesIO(file_bytes), mimetype=mime, resumable=False)
        created = service.files().create(
            body={"name":name,"parents":[day],"description":"ALTIVIA - comprobante original"},
            media_body=media, fields="id,webViewLink", supportsAllDrives=True,
        ).execute()
        fid = created["id"]
        return fid, created.get("webViewLink") or f"https://drive.google.com/file/d/{fid}/view"
    except Exception as exc:
        drive_error("Subir archivo", exc)


def register(member_id, vals, original_bytes, original_name):
    if vals["tipo"] not in TYPES:
        raise ValueError("Tipo inválido")
    if not re.fullmatch(r"[A-Za-z0-9-]{1,12}", vals["serie"]):
        raise ValueError("Verifica la serie")
    if not re.fullmatch(r"[0-9A-Za-z-]{1,20}", vals["numero"]):
        raise ValueError("Verifica el número correlativo")
    if not re.fullmatch(r"\d{11}", vals["ruc"]):
        raise ValueError("El RUC debe contener 11 dígitos")
    try:
        fecha = date.fromisoformat(vals["fecha"])
    except ValueError as exc:
        raise ValueError("Fecha inválida") from exc
    if fecha > date.today():
        raise ValueError("La fecha de emisión no puede ser futura")
    if not 2 <= len(vals["proveedor"]) <= 180:
        raise ValueError("Ingresa la razón social")
    with db() as conn:
        duplicate = conn.execute("SELECT 1 FROM invoices WHERE tipo=? AND ruc=? AND serie=? AND numero=?", (vals["tipo"], vals["ruc"], vals["serie"].upper(), vals["numero"])).fetchone()
        if duplicate:
            raise ValueError("Factura duplicada: ya está registrada")
        member = conn.execute("SELECT 1 FROM members WHERE id=? AND active=1", (member_id,)).fetchone()
        if not member:
            raise ValueError("El colaborador seleccionado ya no está activo")
        # Upload first: do not mark invoice registered if backup failed.
        fid, link = drive_upload(original_bytes, original_name, vals)
        conn.execute("""INSERT INTO invoices(member_id,tipo,serie,numero,fecha,proveedor,ruc,total,created_at,drive_file_id,drive_url,drive_status)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", (
            member_id, vals["tipo"], vals["serie"].upper(), vals["numero"], vals["fecha"],
            vals["proveedor"], vals["ruc"], money(vals["total"]), datetime.now().isoformat(timespec="seconds"),
            fid, link, "Guardado en Drive"
        ))
        conn.commit()


def private_access():
    st.subheader("Acceso administrativo")
    if not st.secrets.get("ADMIN_PASSWORD") or not st.secrets.get("CONTADOR_PASSWORD"):
        st.error("Configura ADMIN_PASSWORD y CONTADOR_PASSWORD en Secrets.")
        return
    if not st.session_state.get("role"):
        with st.form("login"):
            role = st.selectbox("Ingresar como", ["Administrador", "Contador"])
            password = st.text_input("Contraseña", type="password")
            submit = st.form_submit_button("Ingresar", type="primary")
        if submit:
            expected = st.secrets.get("ADMIN_PASSWORD" if role == "Administrador" else "CONTADOR_PASSWORD", "")
            import hmac
            if expected and hmac.compare_digest(password, expected):
                st.session_state.role = role
                st.rerun()
            else:
                st.error("Credenciales incorrectas")
        return
    col1, col2 = st.columns([5,1])
    col1.success("Sesión: " + st.session_state.role)
    if col2.button("Salir"):
        st.session_state.role = None
        st.rerun()
    if st.session_state.role == "Administrador":
        st.markdown("#### Colaboradores")
        with st.form("add_member", clear_on_submit=True):
            name = st.text_input("Nuevo colaborador (nombre completo)")
            save = st.form_submit_button("Agregar miembro")
        if save:
            if not 3 <= len(name.strip()) <= 100:
                st.error("Ingresa un nombre válido")
            else:
                try:
                    with db() as conn:
                        conn.execute("INSERT INTO members(name) VALUES (?)", (name.strip(),))
                        conn.commit()
                    st.success("Miembro agregado")
                except sqlite3.IntegrityError:
                    st.warning("Ese miembro ya existe")
        members = list_members()
        if members:
            st.caption("Miembros activos: " + ", ".join(m["name"] for m in members))
    if st.session_state.role == "Administrador":
        st.markdown("#### Diagnóstico Google Drive")
        if st.button("Comprobar acceso de Google Drive"):
            try:
                folder = drive_check_root(drive_client())
                st.success(f"La cuenta OAuth puede acceder a la carpeta: {folder.get('name', 'Sin nombre')}.")
            except DriveBackupError as exc:
                st.error(str(exc))
    st.markdown("#### Facturas registradas")
    with db() as conn:
        result = pd.read_sql_query("""SELECT i.id AS 'Código',m.name AS 'Colaborador',
           i.tipo AS 'Tipo',i.serie AS 'Serie',i.numero AS 'Número',i.fecha AS 'Fecha emisión',
           i.proveedor AS 'Proveedor',i.ruc AS 'RUC',i.total AS 'Importe',
           i.created_at AS 'Registrada el',i.drive_url AS 'Respaldo en Drive' FROM invoices i JOIN members m ON m.id=i.member_id
           ORDER BY i.id DESC""", conn)
    st.dataframe(result, use_container_width=True, hide_index=True)
    st.download_button("⬇ Exportar CSV", result.to_csv(index=False).encode("utf-8-sig"),
                       file_name="ALTIVIA_facturas.csv", mime="text/csv")


def direct_member_id():
    """Registro sin identificación del colaborador, conservando la FK existente."""
    with db() as conn:
        conn.execute("INSERT OR IGNORE INTO members(name,active) VALUES (?,1)", ("Registro directo",))
        conn.execute("UPDATE members SET active=1 WHERE name=?", ("Registro directo",))
        row = conn.execute("SELECT id FROM members WHERE name=?", ("Registro directo",)).fetchone()
        conn.commit()
        return row["id"]


class UploadedInvoice:
    """Archivo capturado en el componente web, con interfaz compatible con Gemini y Drive."""
    def __init__(self, payload):
        self.name = Path(str(payload.get("filename") or "factura.jpg")).name
        self._data = base64.b64decode(payload["data"], validate=True)
        if not self._data or len(self._data) > 12 * 1024 * 1024:
            raise ValueError("El archivo debe ocupar menos de 12 MB")
        ext = Path(self.name).suffix.lower()
        if ext == ".pdf":
            if not self._data.startswith(b"%PDF"):
                raise ValueError("El PDF no es válido")
        elif ext in (".jpg", ".jpeg", ".png", ".webp"):
            with Image.open(io.BytesIO(self._data)) as im:
                im.verify()
        else:
            raise ValueError("Solo se permiten imágenes JPG, PNG, WEBP o PDF")

    def getvalue(self):
        return self._data


def selected_invoice_document():
    """Ambas tarjetas son controles reales en un componente bidireccional."""
    result = CAMARA_NATIVA(key="altivia_document_cards_v7")
    if not result:
        return None
    try:
        return UploadedInvoice(result)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        st.error(f"No se pudo recibir el archivo: {exc}")
        return None


def public_register():
    st.markdown("""<div class="altivia-register-brand">
        <div class="altivia-register-title">ALTIVIA</div>
        <div class="altivia-register-subtitle">Facturas</div>
    </div>""", unsafe_allow_html=True)
    uploaded = selected_invoice_document()
    if uploaded is None:
        return
    data = uploaded.getvalue()
    stamp = hashlib.sha256(data).hexdigest()
    if st.session_state.get("image_stamp") != stamp:
        st.session_state.image_stamp = stamp
        st.session_state.extracted = dict(SCHEMA)
        st.session_state.analysis_error = None
        with st.spinner("Gemini está analizando la factura automáticamente..."):
            try:
                st.session_state.extracted = recognize(uploaded)
            except Exception as exc:
                st.session_state.analysis_error = str(exc)
    if st.session_state.get("analysis_error"):
        st.warning("No fue posible reconocer todos los datos automáticamente. " + st.session_state.analysis_error + " Puedes completarlos manualmente.")
    else:
        st.success("Lectura automática finalizada. Verifica los datos antes de guardar.")
    vals = st.session_state.extracted
    st.markdown("#### Verificar datos de factura")
    with st.form("invoice_form"):
        c1,c2,c3 = st.columns(3)
        tipo = c1.selectbox("Tipo de comprobante", TYPES, index=TYPES.index(vals["tipo"]) if vals["tipo"] in TYPES else 0)
        serie = c2.text_input("Serie", vals["serie"], placeholder="F001")
        numero = c3.text_input("Número correlativo", vals["numero"], placeholder="000123")
        c4,c5,c6 = st.columns(3)
        try:
            parsed_date = date.fromisoformat(vals["fecha"])
        except ValueError:
            parsed_date = None
        fecha = c4.date_input("Fecha de emisión", value=parsed_date, format="DD/MM/YYYY", min_value=date(2000,1,1))
        proveedor = c5.text_input("Proveedor / razón social", vals["proveedor"])
        ruc = c6.text_input("RUC del proveedor", vals["ruc"], max_chars=11)
        total = st.text_input("Importe total", vals["total"], placeholder="150.00")
        submit = st.form_submit_button("✓ Confirmar y guardar factura", type="primary")
    if submit:
        if not fecha:
            st.error("Ingresa la fecha de emisión")
        else:
            try:
                register(direct_member_id(), {"tipo":tipo, "serie":serie.strip(),
                          "numero":numero.strip(), "fecha":fecha.isoformat(),
                          "proveedor":proveedor.strip(), "ruc":ruc.strip(), "total":total.strip()},
                          data, uploaded.name)
                st.success("Factura registrada y documento respaldado en Google Drive.")
            except sqlite3.IntegrityError:
                st.error("Factura duplicada: ya existe una con ese tipo, RUC, serie y número.")
            except ValueError as exc:
                st.error(str(exc))
            except DriveBackupError as exc:
                st.error(f"No se guardó el registro. {exc}")
            except Exception as exc:
                import logging
                logging.exception("ALTIVIA: fallo inesperado durante el registro")
                st.error(f"No se guardó el registro. Error inesperado: {type(exc).__name__}. Revisa los logs privados.")


st.markdown("""<style>
.block-container{max-width:1250px;padding-top:1.25rem}
.altivia-register-brand{margin:1rem 0 1.45rem;text-align:left}
.altivia-register-title{font-size:clamp(2.3rem,5vw,3.5rem);font-weight:850;color:#102c4b;letter-spacing:.055em;line-height:1.12}
.altivia-register-subtitle{font-size:clamp(1.35rem,3vw,1.95rem);font-weight:550;color:#1767ab;margin-top:3px}
@media(max-width:590px){.altivia-register-brand{margin:.7rem 0 1.1rem}}
</style>""", unsafe_allow_html=True)
# La pantalla principal ES el registro: no existe una bienvenida ni una navegación intermedia.
_, navright = st.columns([4, 1])
if navright.button("Ingreso administrativo", use_container_width=True):
    st.session_state.screen = "admin"
    st.rerun()
page = st.session_state.get("screen", "register")
if page == "admin":
    if st.button("← Volver a facturas", key="back_to_registration"):
        st.session_state.screen = "register"
        st.rerun()
    private_access()
else:
    public_register()
