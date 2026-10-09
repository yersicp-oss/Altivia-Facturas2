"""ALTIVIA Facturas Streamlit Lite — versión prototipo sin almacenamiento persistente."""
import base64
import hashlib
import io
import json
import re
import sqlite3
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI
from PIL import Image, ImageOps, UnidentifiedImageError

st.set_page_config(page_title="ALTIVIA | Facturas", page_icon="🧾", layout="wide")
DB_PATH = Path("altivia_demo.db")
TYPES = ["Factura", "Boleta", "Nota de crédito", "Otro"]
SCHEMA = {"tipo": "", "serie": "", "numero": "", "fecha": "", "proveedor": "", "ruc": "", "total": ""}

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
    key = st.secrets.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("Falta OPENAI_API_KEY en los Secrets de Streamlit")
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
    payload = [{"type": "text", "text": (
        "Lee el comprobante peruano de la(s) imagen(es). Responde SOLO JSON válido con "
        "estas siete claves exactas: tipo, serie, numero, fecha, proveedor, ruc, total. "
        "Tipo: Factura/Boleta/Nota de crédito/Otro. fecha: AAAA-MM-DD. "
        "Total: valor numérico sin moneda ni símbolos, punto decimal. "
        "Número correlativo y serie van separados. RUC es exactamente 11 dígitos si aparece. "
        "No inventes datos; si un dato no está legible responde cadena vacía. "
        "Si hay varios importes elige solamente el importe total a pagar."
    )}]
    for img in pictures:
        data_url = "data:image/jpeg;base64," + base64.b64encode(img).decode("ascii")
        payload.append({"type": "image_url", "image_url": {"url": data_url, "detail": "low"}})
    client = OpenAI(api_key=key, timeout=23.0, max_retries=0)
    response = client.chat.completions.create(
        model="gpt-4.1-mini", messages=[{"role": "user", "content": payload}],
        temperature=0, response_format={"type": "json_object"}, max_tokens=280
    )
    return parse_json(response.choices[0].message.content or "{}")


def money(v):
    try:
        n = float(str(v).replace(",", "."))
    except ValueError as exc:
        raise ValueError("El importe debe ser numérico") from exc
    if not 0 < n < 1_000_000_000:
        raise ValueError("Importe total fuera de rango")
    return n


def register(member_id, vals):
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
        member = conn.execute("SELECT 1 FROM members WHERE id=? AND active=1", (member_id,)).fetchone()
        if not member:
            raise ValueError("El colaborador seleccionado ya no está activo")
        conn.execute("""INSERT INTO invoices(member_id,tipo,serie,numero,fecha,proveedor,ruc,total,created_at)
                        VALUES (?,?,?,?,?,?,?,?,?)""", (
            member_id, vals["tipo"], vals["serie"].upper(), vals["numero"], vals["fecha"],
            vals["proveedor"], vals["ruc"], money(vals["total"]), datetime.now().isoformat(timespec="seconds")
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
    st.markdown("#### Facturas registradas")
    with db() as conn:
        result = pd.read_sql_query("""SELECT i.id AS 'Código',m.name AS 'Colaborador',
           i.tipo AS 'Tipo',i.serie AS 'Serie',i.numero AS 'Número',i.fecha AS 'Fecha emisión',
           i.proveedor AS 'Proveedor',i.ruc AS 'RUC',i.total AS 'Importe',
           i.created_at AS 'Registrada el' FROM invoices i JOIN members m ON m.id=i.member_id
           ORDER BY i.id DESC""", conn)
    st.dataframe(result, use_container_width=True, hide_index=True)
    st.download_button("⬇ Exportar CSV", result.to_csv(index=False).encode("utf-8-sig"),
                       file_name="ALTIVIA_facturas.csv", mime="text/csv")


def public_register():
    st.subheader("Registrar factura")
    st.caption("Selecciona tu nombre, sube una foto o PDF y confirma los datos reconocidos.")
    members = list_members()
    if not members:
        st.warning("El administrador debe registrar primero a los miembros.")
        return
    member_by_name = {m["name"]: m["id"] for m in members}
    member = st.selectbox("¿Quién registra la factura?", list(member_by_name))
    mode = st.radio("Documento", ["Tomar foto", "Subir imagen", "Subir PDF"], horizontal=True)
    if mode == "Tomar foto":
        uploaded = st.camera_input("Tomar fotografía")
    elif mode == "Subir imagen":
        uploaded = st.file_uploader("Selecciona imagen", type=["jpg", "jpeg", "png", "webp"])
    else:
        uploaded = st.file_uploader("Selecciona PDF", type=["pdf"])
    if uploaded:
        stamp = hashlib.sha256(uploaded.getvalue()).hexdigest()
        if st.session_state.get("image_stamp") != stamp:
            st.session_state.image_stamp = stamp
            st.session_state.extracted = dict(SCHEMA)
        st.caption("El archivo será enviado al servicio de IA para sugerir los datos. No se guardará una copia documental.")
        if st.button("✨ Leer factura con IA", type="primary"):
            with st.spinner("La IA está leyendo el comprobante..."):
                try:
                    st.session_state.extracted = recognize(uploaded)
                    st.success("Lectura finalizada. Revisa los campos.")
                except Exception as exc:
                    st.warning(f"No se completó la lectura automática: {exc}")
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
                    register(member_by_name[member], {"tipo":tipo, "serie":serie.strip(),
                              "numero":numero.strip(), "fecha":fecha.isoformat(),
                              "proveedor":proveedor.strip(), "ruc":ruc.strip(), "total":total.strip()})
                    st.success("Factura registrada. El administrador puede verla en su panel.")
                except sqlite3.IntegrityError:
                    st.error("Factura duplicada: ya existe una con ese tipo, RUC, serie y número.")
                except ValueError as exc:
                    st.error(str(exc))


st.title("A. ALTIVIA  |  Gestión de facturas")
st.caption("Versión ligera para Streamlit · Reconocimiento visual con IA, sin Tesseract")
st.info("Prototipo de pruebas: el servidor gratuito no garantiza conservar los registros. No cargar comprobantes reales hasta conectar almacenamiento permanente.", icon="⚠️")
mode = st.sidebar.radio("Menú", ["Registrar factura", "Acceso privado"])
if mode == "Registrar factura":
    public_register()
else:
    private_access()
