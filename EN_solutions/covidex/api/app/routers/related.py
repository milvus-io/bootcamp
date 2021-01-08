import json
from datetime import datetime
from typing import List
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request

from app.models import (RelatedArticle, RelatedQueryResponse, SearchLogData,
                        SearchLogType, SearchVertical)
from app.settings import settings
from app.util.logging import build_timed_logger
from app.util.request import get_doc_url, get_multivalued_field, get_request_ip

router = APIRouter()
related_logger = build_timed_logger('related_logger', 'related.log')


@router.get('/related/{uid}', response_model=RelatedQueryResponse)
async def get_related(request: Request, uid: str, page_number: int = 1, query_id: str = None):
    searcher = request.app.state.searcher
    related_searcher = request.app.state.related_searcher

    # Invalid uid -> 404
    if uid not in related_searcher.uid_set:
        raise HTTPException(status_code=404, detail="Item not found")

    source_vector = related_searcher.embedding[uid]
    related_results = []

    source_vector = [float(vec) for vec in source_vector]

    parameters = {'ef': 101}
    # print(related_searcher.milvus.count_entities(related_searcher.collection_name))
    status, results = related_searcher.milvus.search(collection_name=related_searcher.collection_name, query_records=[source_vector], top_k=100, params=parameters)
    print("Milvus search.", status)
    labels = results.id_array
    distances = results.distance_array
    # print(len(labels), len(distances))

    start_idx = (page_number - 1)*20
    end_idx = start_idx + 20
    for index, dist in zip(labels[0][start_idx:end_idx], distances[0][start_idx:end_idx]):
        uid = related_searcher.index_to_uid[index]
        hit = searcher.doc(uid, SearchVertical.cord19)
        if hit.lucene_document() is None:
            continue
        result = build_related_result(hit, uid, dist)
        related_results.append(result)

    # Generate UUID for query.
    query_id = str(uuid4())

    # Log query and results.
    related_logger.info(json.dumps({
        'query_id': query_id,
        'uid': uid,
        'page_number': page_number,
        'request_ip': get_request_ip(request),
        'timestamp': datetime.utcnow().isoformat(),
        'response': [r.json() for r in related_results],
    }))

    return RelatedQueryResponse(query_id=query_id, response=related_results)


@router.post('/related/log/clicked', response_model=None)
async def post_clicked(data: SearchLogData):
    related_logger.info(json.dumps({
        'query_id': data.query_id,
        'type': SearchLogType.clicked,
        'result_id': data.result_id,
        'position': data.position,
        'timestamp': datetime.utcnow().isoformat()}))


def build_related_result(hit, id: str, dist: float):
    doc = hit.lucene_document()
    return RelatedArticle(
        id=id,
        abstract=doc.get('abstract'),
        authors=get_multivalued_field(doc, 'authors'),
        distance=dist,
        journal=doc.get('journal'),
        publish_time=doc.get('publish_time'),
        source=get_multivalued_field(doc, 'source_x'),
        title=doc.get('title'),
        url=get_doc_url(doc)
    )
