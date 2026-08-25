from fastapi.testclient import TestClient

from web.backend.app import app


def test_web_health_and_index() -> None:
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        payload = health.json()
        assert payload["status"] == "ok"
        assert payload["research_only"] is True
        assert "inference_ready" in payload

        index = client.get("/")
        assert index.status_code == 200
        assert "text/html" in index.headers.get("content-type", "")
        assert "骨科 CT 智能辅助分析研究平台" in index.text
