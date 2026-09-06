from fastapi.testclient import TestClient
from api.main import app


client = TestClient(app)


def test_pipeline_status_healthy():
    response = client.get("/pipeline/status")

    assert response.status_code == 200

    data = response.json()

    assert data["healthy"] is True
    assert data["reasons"] == []
    assert data["status"] == "SUCCESS"
    assert data["as_of"] == "2013-11-08T04:30:00.000+05:30"


def test_pipeline_status_unhealthy_injected_failure(monkeypatch):
    import api.main as main

    original_loader = main.load_pipeline_status_record

    def fake_status_record():
        record = original_loader().copy()

        record["status"] = "FAILED"
        record["tasks"] = {
            "ingest": "SUCCESS",
            "validate": "FAILED",
            "spark_process": "SUCCESS",
            "load_warehouse": "SUCCESS",
            "quality_check": "SUCCESS",
            "notify": "PENDING",
        }

        return record

    monkeypatch.setattr(
        main,
        "load_pipeline_status_record",
        fake_status_record,
    )

    response = client.get("/pipeline/status")

    assert response.status_code == 200

    data = response.json()

    assert data["healthy"] is False
    assert len(data["reasons"]) > 0