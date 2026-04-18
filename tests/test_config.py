import pytest
from unittest.mock import patch
from app.core.config import Settings, get_settings


def test_settings_load_from_env():
    with patch.dict("os.environ", {
        "GROQ_API_KEY": "test-key-123",
        "MAX_UPLOAD_MB": "10",
        "CHAT_MODEL": "llama-3.1-8b-instant",
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "CHROMA_DB_PATH": "./chroma_db",
    }):
        s = Settings(_env_file=None)
        assert s.groq_api_key == "test-key-123"
        assert s.max_upload_mb == 10
        assert s.chat_model == "llama-3.1-8b-instant"
        assert s.embedding_model == "all-MiniLM-L6-v2"
        assert s.chroma_db_path == "./chroma_db"


def test_settings_defaults():
    with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
        s = Settings(_env_file=None)
        assert s.max_upload_mb == 5
        assert s.embedding_model == "all-MiniLM-L6-v2"
        assert s.chat_model == "llama-3.1-8b-instant"
        assert s.chroma_db_path == "./chroma_db"


def test_missing_groq_key_raises():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(Exception):
            Settings(_env_file=None)


def test_get_settings_returns_cached_instance():
    get_settings.cache_clear()
    with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
    get_settings.cache_clear()
