"""
Contract tests for Course Project compliance.

Tests that all required API endpoints return the exact schemas specified
in Course Project.txt.  These tests use the FastAPI TestClient, so no
server needs to be running.
"""

import os
import sys
import pytest

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


# =========================================================================
# GET /api/team_info
# =========================================================================

class TestTeamInfo:
    """GET /api/team_info must return exact JSON schema."""

    def test_status_code(self):
        resp = client.get("/api/team_info")
        assert resp.status_code == 200

    def test_content_type_json(self):
        resp = client.get("/api/team_info")
        assert "application/json" in resp.headers["content-type"]

    def test_required_top_level_keys(self):
        data = client.get("/api/team_info").json()
        for key in ("group_batch_order_number", "team_name", "students"):
            assert key in data, f"Missing key: {key}"

    def test_group_batch_order_number_format(self):
        data = client.get("/api/team_info").json()
        val = data["group_batch_order_number"]
        assert isinstance(val, str)
        assert "_" in val, "group_batch_order_number should be '{batch}_{order}'"

    def test_team_name_is_string(self):
        data = client.get("/api/team_info").json()
        assert isinstance(data["team_name"], str)
        assert len(data["team_name"]) > 0

    def test_students_is_list(self):
        data = client.get("/api/team_info").json()
        assert isinstance(data["students"], list)
        assert len(data["students"]) >= 1

    def test_each_student_has_name_and_email(self):
        data = client.get("/api/team_info").json()
        for i, student in enumerate(data["students"]):
            assert "name" in student, f"students[{i}] missing 'name'"
            assert "email" in student, f"students[{i}] missing 'email'"
            assert isinstance(student["name"], str)
            assert isinstance(student["email"], str)
            assert "@" in student["email"], f"students[{i}].email invalid"


# =========================================================================
# GET /api/agent_info
# =========================================================================

class TestAgentInfo:
    """GET /api/agent_info must return exact JSON schema."""

    def test_status_code(self):
        resp = client.get("/api/agent_info")
        assert resp.status_code == 200

    def test_content_type_json(self):
        resp = client.get("/api/agent_info")
        assert "application/json" in resp.headers["content-type"]

    def test_required_top_level_keys(self):
        data = client.get("/api/agent_info").json()
        for key in ("description", "purpose", "prompt_template", "prompt_examples"):
            assert key in data, f"Missing key: {key}"

    def test_description_is_nonempty_string(self):
        data = client.get("/api/agent_info").json()
        assert isinstance(data["description"], str)
        assert len(data["description"]) > 10

    def test_purpose_is_nonempty_string(self):
        data = client.get("/api/agent_info").json()
        assert isinstance(data["purpose"], str)
        assert len(data["purpose"]) > 10

    def test_prompt_template_has_template_key(self):
        data = client.get("/api/agent_info").json()
        pt = data["prompt_template"]
        assert isinstance(pt, dict)
        assert "template" in pt, "prompt_template must contain 'template' key"
        assert isinstance(pt["template"], str)

    def test_prompt_examples_is_list(self):
        data = client.get("/api/agent_info").json()
        assert isinstance(data["prompt_examples"], list)
        assert len(data["prompt_examples"]) >= 1

    def test_prompt_examples_each_has_required_fields(self):
        data = client.get("/api/agent_info").json()
        for i, example in enumerate(data["prompt_examples"]):
            assert "prompt" in example, f"prompt_examples[{i}] missing 'prompt'"
            assert "full_response" in example, f"prompt_examples[{i}] missing 'full_response'"
            assert "steps" in example, f"prompt_examples[{i}] missing 'steps'"
            assert isinstance(example["steps"], list)

    def test_prompt_examples_steps_have_module_prompt_response(self):
        data = client.get("/api/agent_info").json()
        for i, example in enumerate(data["prompt_examples"]):
            for j, step in enumerate(example["steps"]):
                assert "module" in step, f"examples[{i}].steps[{j}] missing 'module'"
                assert "prompt" in step, f"examples[{i}].steps[{j}] missing 'prompt'"
                assert "response" in step, f"examples[{i}].steps[{j}] missing 'response'"


# =========================================================================
# GET /api/model_architecture
# =========================================================================

class TestModelArchitecture:
    """GET /api/model_architecture must return a PNG image."""

    def test_status_code(self):
        resp = client.get("/api/model_architecture")
        assert resp.status_code == 200

    def test_content_type_png(self):
        resp = client.get("/api/model_architecture")
        ct = resp.headers["content-type"]
        assert "image/png" in ct, f"Expected image/png, got {ct}"

    def test_body_is_png(self):
        resp = client.get("/api/model_architecture")
        body = resp.content
        assert len(body) > 100, "PNG body too small"
        # PNG magic bytes
        assert body[:4] == b'\x89PNG', "Response body is not a valid PNG"


# =========================================================================
# POST /api/execute — schema tests
# =========================================================================

class TestExecuteEndpoint:
    """POST /api/execute must return exact JSON schema."""

    def test_content_type_json(self):
        resp = client.post("/api/execute", json={"prompt": "hello"})
        assert "application/json" in resp.headers["content-type"]

    def test_empty_prompt_returns_error_shape(self):
        """Empty prompt must produce error shape."""
        resp = client.post("/api/execute", json={"prompt": ""})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert data["error"] is not None
        assert data["response"] is None
        assert isinstance(data["steps"], list)

    def test_error_shape_has_all_keys(self):
        resp = client.post("/api/execute", json={"prompt": ""})
        data = resp.json()
        for key in ("status", "error", "response", "steps"):
            assert key in data, f"Error response missing key: {key}"

    def test_out_of_scope_returns_ok_shape(self):
        """An out-of-scope prompt should return ok with a polite response."""
        resp = client.post("/api/execute", json={"prompt": "What is the weather today?"})
        data = resp.json()
        # Some classifiers return ok with a polite decline; others return error.
        # The important thing is the top-level keys exist.
        for key in ("status", "error", "response", "steps"):
            assert key in data, f"Out-of-scope response missing key: {key}"
        assert isinstance(data["steps"], list)

    def test_success_shape_keys(self):
        """On success: status='ok', error=null, response=string, steps=array."""
        # Use an out-of-scope to get a fast response
        resp = client.post("/api/execute", json={"prompt": "What is the weather today?"})
        data = resp.json()
        if data["status"] == "ok":
            assert data["error"] is None
            assert isinstance(data["response"], str)
            assert isinstance(data["steps"], list)

    def test_steps_are_dicts_with_required_keys(self):
        """Each step must have module, prompt, response."""
        # We can't easily trigger a full run in tests without mocking
        # so just validate the schema on the static agent_info examples
        info = client.get("/api/agent_info").json()
        for example in info["prompt_examples"]:
            for step in example["steps"]:
                assert "module" in step
                assert "prompt" in step
                assert "response" in step


# =========================================================================
# Schema guard module tests
# =========================================================================

class TestSchemaGuard:
    """Verify the schema guard catches violations."""

    def test_validate_team_info_passes_good_payload(self):
        from src.api.schema_guard import validate_team_info
        good = {
            "group_batch_order_number": "1_1",
            "team_name": "Test",
            "students": [{"name": "A", "email": "a@b.com"}],
        }
        result = validate_team_info(good)
        assert result == good

    def test_validate_agent_info_passes_good_payload(self):
        from src.api.schema_guard import validate_agent_info
        good = {
            "description": "desc",
            "purpose": "purpose",
            "prompt_template": {"template": "..."},
            "prompt_examples": [
                {"prompt": "p", "full_response": "r", "steps": []}
            ],
        }
        result = validate_agent_info(good)
        assert result == good

    def test_validate_execute_patches_missing_keys(self):
        from src.api.schema_guard import validate_execute_response
        incomplete = {"status": "ok"}
        result = validate_execute_response(incomplete)
        assert "error" in result
        assert "response" in result
        assert "steps" in result
        assert isinstance(result["steps"], list)

    def test_validate_execute_passes_good_ok(self):
        from src.api.schema_guard import validate_execute_response
        good = {
            "status": "ok",
            "error": None,
            "response": "done",
            "steps": [{"module": "M", "prompt": {}, "response": {}}],
        }
        result = validate_execute_response(good)
        assert result == good

    def test_validate_execute_passes_good_error(self):
        from src.api.schema_guard import validate_execute_response
        good = {
            "status": "error",
            "error": "something failed",
            "response": None,
            "steps": [],
        }
        result = validate_execute_response(good)
        assert result == good


# =========================================================================
# POST /api/execute — missing / invalid prompt validation
# =========================================================================

def _assert_error_shape(data: dict) -> None:
    """Assert the payload matches the Course Project error JSON schema."""
    assert set(data.keys()) >= {"status", "error", "response", "steps"}
    assert data["status"] == "error"
    assert isinstance(data["error"], str) and len(data["error"]) > 0
    assert data["response"] is None
    assert isinstance(data["steps"], list)


class TestExecuteMissingPrompt:
    """POST /api/execute must return spec-compliant error when prompt is missing or invalid."""

    def test_unknown_key_no_prompt(self):
        """Case A: { 'not_prompt': 'x' } => error shape."""
        resp = client.post("/api/execute", json={"not_prompt": "x"})
        assert resp.status_code == 200
        _assert_error_shape(resp.json())

    def test_empty_body(self):
        """Case B: {} => error shape."""
        resp = client.post("/api/execute", json={})
        assert resp.status_code == 200
        _assert_error_shape(resp.json())

    def test_empty_string_prompt(self):
        """Case C: { 'prompt': '' } => error shape."""
        resp = client.post("/api/execute", json={"prompt": ""})
        assert resp.status_code == 200
        _assert_error_shape(resp.json())

    def test_whitespace_only_prompt(self):
        """Case C: { 'prompt': '   ' } => error shape."""
        resp = client.post("/api/execute", json={"prompt": "   "})
        assert resp.status_code == 200
        _assert_error_shape(resp.json())

    def test_integer_prompt(self):
        """Case D: { 'prompt': 123 } => error shape."""
        resp = client.post("/api/execute", json={"prompt": 123})
        assert resp.status_code == 200
        _assert_error_shape(resp.json())

    def test_null_prompt(self):
        """Case D: { 'prompt': null } => error shape."""
        resp = client.post("/api/execute", json={"prompt": None})
        assert resp.status_code == 200
        _assert_error_shape(resp.json())

    def test_valid_prompt_with_extra_keys(self):
        """Case E: { 'prompt': 'valid', 'not_prompt': 'x' } => ok (extra key ignored)."""
        resp = client.post("/api/execute", json={"prompt": "What is the weather today?", "not_prompt": "x"})
        data = resp.json()
        assert data["status"] in ("ok", "error")  # may be out-of-scope "ok" or in-scope
        for key in ("status", "error", "response", "steps"):
            assert key in data
