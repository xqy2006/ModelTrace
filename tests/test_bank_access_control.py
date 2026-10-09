import pytest

import app as app_module

REMOTE_ADDR = "192.168.1.100"
LOCAL_ADDRS = ("127.0.0.1", "::1")

PROTECTED_ROUTES = [
    ("GET", "/api/bank"),
    ("GET", "/api/banks"),
    ("POST", "/api/banks"),
    ("POST", "/api/enroll/auto"),
]


@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
def test_bank_endpoint_rejects_remote_client(client, method, path):
    request_method = getattr(client, method.lower())
    response = request_method(path, environ_base={"REMOTE_ADDR": REMOTE_ADDR})
    assert response.status_code == 403


@pytest.mark.parametrize("addr", LOCAL_ADDRS)
def test_get_bank_allows_local_client(client, addr):
    response = client.get("/api/bank", environ_base={"REMOTE_ADDR": addr})
    assert response.status_code != 403


@pytest.mark.parametrize("addr", LOCAL_ADDRS)
def test_get_banks_allows_local_client(client, addr):
    response = client.get("/api/banks", environ_base={"REMOTE_ADDR": addr})
    assert response.status_code != 403


def test_create_bank_allows_local_client(client, monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "save_custom_configs", lambda: None)
    monkeypatch.setattr(app_module, "PROJECT", tmp_path)

    bank_id = "pytest-boundary-check"
    assert bank_id not in app_module.BANK_CONFIGS
    try:
        response = client.post(
            "/api/banks",
            json={"label": "Pytest Boundary Check"},
            environ_base={"REMOTE_ADDR": "127.0.0.1"},
        )
        assert response.status_code != 403
    finally:
        app_module.BANK_CONFIGS.pop(bank_id, None)
        app_module.banks.pop(bank_id, None)


def test_enroll_auto_allows_local_client(client, monkeypatch):
    monkeypatch.setattr(app_module, "enroll_automatic", lambda **kwargs: {})
    monkeypatch.setattr(app_module, "replace_bank", lambda bank_id: None)

    response = client.post(
        "/api/enroll/auto",
        json={
            "base_url": "https://example.invalid",
            "api_key": "test-key",
            "api_model": "test-model",
            "model_label": "Test Model",
        },
        environ_base={"REMOTE_ADDR": "127.0.0.1"},
    )
    assert response.status_code != 403
