"""Ejecuta localmente. NUNCA subas token.json ni credenciales a GitHub.
1) pip install google-auth-oauthlib
2) Descarga OAuth client tipo Desktop como client_secret.json junto a este archivo.
3) python CONFIGURAR_DRIVE_LOCAL.py
4) Copia el refresh token y client id/secret a Streamlit Secrets.
"""
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow

path = Path('client_secret.json')
if not path.exists():
    raise SystemExit('Coloca client_secret.json (OAuth client tipo Desktop) en esta carpeta.')
flow = InstalledAppFlow.from_client_secrets_file(str(path), scopes=['https://www.googleapis.com/auth/drive'])
creds = flow.run_local_server(port=0, access_type='offline', prompt='consent')
client_data = __import__('json').loads(path.read_text(encoding='utf-8'))
client = client_data.get('installed') or client_data.get('web')
print('\nCopia estas variables A STREAMLIT SECRETS, NUNCA A GITHUB:\n')
print('GOOGLE_CLIENT_ID = ' + repr(client['client_id']))
print('GOOGLE_CLIENT_SECRET = ' + repr(client['client_secret']))
print('GOOGLE_REFRESH_TOKEN = ' + repr(creds.refresh_token))
print('\nAdvertencia: estos valores son secretos. No compartas esta pantalla.\n')
