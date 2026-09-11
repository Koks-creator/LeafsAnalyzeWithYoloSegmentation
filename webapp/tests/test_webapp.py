import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import base64
from io import BytesIO
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image

from webapp import app

"""
mock_post.called              # bool: czy w ogóle był wołany (== call_count > 0)
mock_post.call_args           # argumenty OSTATNIEGO wywołania
mock_post.call_args_list      # lista argumentów WSZYSTKICH wywołań po kolei
mock_post.assert_called_once()           # twierdzi, że było dokładnie 1 wywołanie
mock_post.assert_not_called()            # twierdzi, że było 0 wywołań
mock_post.assert_called_with(...)        # twierdzi, że ostatnie wołanie miało takie argumenty
"""
def _fake_b64_image():
    img = Image.new("RGB", (8, 8), color="red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

@pytest.fixture()
def app_client():
    app.config.update({
        "TESTING": True,
        "WTF_CSRF_ENABLED": False,
    })
    yield app

def test_home_page(app_client):
    response = app_client.test_client().get("/")

    assert response.status_code == 200

def test_health_when_api_up(app_client, monkeypatch):
    # faking api
    monkeypatch.setattr("webapp.routes.check_api_connection", lambda: True)
    resp = app_client.test_client().get("/health")
    assert resp.json == {"status": True}

def test_health_when_api_down(app_client, monkeypatch):
    monkeypatch.setattr("webapp.routes.check_api_connection", lambda: False)
    resp = app_client.test_client().get("/health")
    assert resp.json == {"status": False}


# mock_check ← @patch(...check_api_connection...)
# mock_post ← @patch(...requests.post...)
@patch("webapp.routes.requests.post")
@patch("webapp.routes.check_api_connection", return_value=True)
def test_gen_num_out_of_range(mock_check, mock_post, app_client):
    client = app_client.test_client()
    resp = client.post("/", data={"gen_num_field": 100, "models_list_field": "dcgan"})

    assert resp.status_code == 200
    mock_post.assert_not_called()
    assert b"face.png" not in resp.data

@patch("webapp.routes.requests.post")
@patch("webapp.routes.check_api_connection", return_value=True)
def test_invalid_model(mock_check, mock_post, app_client):
    client = app_client.test_client()
    resp = client.post("/", data={"gen_num_field": 2, "models_list_field": "dupen"})

    assert resp.status_code == 200
    mock_post.assert_not_called()
    assert b"face.png" not in resp.data


@patch("webapp.routes.cv2.imwrite")
@patch("webapp.routes.requests.post")
@patch("webapp.routes.check_api_connection", return_value=True)
def test_generate_faces_happy_path(mock_check, mock_post, mock_imwrite, app_client):
    token_resp = MagicMock(status_code=200)
    token_resp.json.return_value = {"access_token": "abc"}

    gen_resp = MagicMock(status_code=200)
    gen_resp.json.return_value = {"images": [_fake_b64_image(), _fake_b64_image()]}

    mock_post.side_effect = [token_resp, gen_resp]

    client = app_client.test_client()
    resp = client.post("/", data={"gen_num_field": 2, "models_list_field": "dcgan"})

    assert resp.status_code == 200
    assert mock_post.call_count == 2
    assert mock_imwrite.call_count == 2
    assert b"face.png" in resp.data

@patch("webapp.routes.requests.post")
@patch("webapp.routes.check_api_connection", return_value=True)
def test_token_request_fails(mock_check, mock_post, app_client):
    mock_post.return_value = MagicMock(status_code=401, content=b"unauthorized")
    mock_post.return_value.json.value = {}

    client = app_client.test_client()
    resp = client.post("/", data={"gen_num_field": 2, "models_list_field": "dcgan"})

    assert resp.status_code == 200
    assert mock_post.call_count == 1
    assert b"face.png" not in resp.data

@patch("webapp.routes.requests.post")
@patch("webapp.routes.check_api_connection", return_value=True)
def test_generate_request_fails(mock_check, mock_post, app_client):
    token_resp = MagicMock(status_code=200)
    token_resp.json.return_value = {"access_token": "abc"}

    face_gen_resp = MagicMock(status_code=500, content=b'jeblowchuj')

    mock_post.side_effect = [token_resp, face_gen_resp]

    client = app_client.test_client()
    resp = client.post("/", data={"gen_num_field": 2, "models_list_field": "dcgan"})

    assert resp.status_code == 200
    assert b"face.png" not in resp.data