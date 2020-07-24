from typing import List

from pydantic import BaseModel
from pydantic.class_validators import validator
from enum import Enum


class BaseArticle(BaseModel):
    id: str
    abstract: str = None
    authors: List[str] = []
    journal: str = None
    publish_time: str = None
    title: str
    source: List[str] = []
    url: str


class SearchArticle(BaseArticle):
    score: float
    paragraphs: List[str] = []
    paragraphs: List[str] = []
    highlights: List[List[tuple]] = []
    highlighted_abstract: bool = False
    has_related_articles: bool = False

    @validator('highlights')
    def validate_highlights(cls, v, values):
        if v:
            assert len(v) == len(values['paragraphs'])
        return v


class RelatedArticle(BaseArticle):
    distance: float


class SearchQueryResponse(BaseModel):
    query_id: str
    response: List[SearchArticle]


class RelatedQueryResponse(BaseModel):
    query_id: str
    response: List[RelatedArticle]


class SearchLogData(BaseModel):
    query_id: str
    result_id: str
    position: int


class SearchLogType(str, Enum):
    query = 'query'
    collapsed = 'collapsed'
    expanded = 'expanded'
    clicked = 'clicked'


class SearchVertical(str, Enum):
    cord19 = 'cord19'
    trialstreamer = 'trialstreamer'
