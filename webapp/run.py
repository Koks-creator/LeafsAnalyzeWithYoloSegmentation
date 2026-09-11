import sys
from pathlib import Path
import ssl
sys.path.append(str(Path(__file__).resolve().parent.parent))

from webapp import app
from config import Config

def run() -> None:
    if Config.WEB_APP_USE_SSL:
        context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        context.load_cert_chain(f"{Config.WEB_APP_SSL_FOLDER}/cert.pem", f"{Config.WEB_APP_SSL_FOLDER}/key.pem")
        app.run(host=Config.WEB_APP_HOST, port=Config.WEB_APP_PORT, debug=Config.WEB_APP_DEBUG, ssl_context=context)
    else:
        app.run(host=Config.WEB_APP_HOST, port=Config.WEB_APP_PORT, debug=Config.WEB_APP_DEBUG)


if __name__ == "__main__":
    run()