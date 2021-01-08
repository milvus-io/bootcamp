from typing import List

from fastapi import Request


def get_request_ip(request: Request):
    forwarded_header = 'X-Forwarded-For'
    request_ip = request.client.host
    if forwarded_header in request.headers:
        request_ip = request.headers[forwarded_header]

    return request_ip


def get_doc_url(doc) -> str:
    doi = doc.get('doi')
    if doi:
        return f'https://doi.org/{doi}'
    return doc.get('url')

def get_multivalued_field(doc, field) -> List[str]:
    return [field.stringValue() for field in doc.getFields(field)]
