import datetime
import logging
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import SearchVertical
from app.services.highlighter import Highlighter
from app.services.ranker import Ranker
from app.services.related_searcher import RelatedSearcher
from app.services.searcher import Searcher


@pytest.fixture(scope="module")
def client():
    '''Client for testing endpoints'''
    app.state.highlighter = MagicMock(Highlighter)
    app.state.ranker = MagicMock(Ranker)
    app.state.related_searcher = MagicMock(RelatedSearcher)
    app.state.searcher = MagicMock(Searcher)

    client = TestClient(app)
    yield client
