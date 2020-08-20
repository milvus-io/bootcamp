import React, { useState, useEffect } from 'react';
import { useRouteMatch } from 'react-router';
import styled from 'styled-components';
import ErrorBoundary from 'react-error-boundary';
import InfiniteScroll from 'react-infinite-scroller';

import {
  Heading2,
  Heading3,
  PageContent,
  PageWrapper,
  Body,
  BodySmall,
  LinkStyle,
} from '../../../shared/Styles';
import { RelatedArticle } from '../../../shared/Models';
import { API_BASE, RELATED_ENDPOINT, HOME_ROUTE } from '../../../shared/Constants';
import Loading from '../../common/Loading';
import RelatedResult from './RelatedResult';
import { FileText, ArrowLeft } from 'react-feather';
import BaseArticleResult from '../../common/BaseArticleResult';
import { parseAbstract } from '../../../shared/Util';
import { Link as RouterLink } from 'react-router-dom';

const NotFoundComponent = () => <NotFound>Article not found</NotFound>;

const RelatedPage = () => {
  const {
    params: { articleId },
  } = useRouteMatch<any>();

  const [loading, setLoading] = useState<boolean>(false);
  const [notFound, setNotFound] = useState<boolean>(false);
  const [hasMore, setHasMore] = useState<boolean>(true);
  const [page, setPage] = useState<number>(1);
  const [queryId, setQueryId] = useState<string>('');

  const [originalArticle, setOriginalArticle] = useState<RelatedArticle | null>(null);
  const [relatedArticles, setRelatedArticles] = useState<RelatedArticle[] | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      if (articleId === undefined || articleId === null || articleId === '') {
        setLoading(false);
        setNotFound(true);
        setPage(1);
        return;
      }

      try {
        setLoading(true);
        setRelatedArticles(null);

        let response = await fetch(
          `${API_BASE}${RELATED_ENDPOINT}/${articleId.toLowerCase()}?page_number=${1}`,
        );
        setLoading(false);

        let data = await response.json();
        const { query_id, response: responseArticles } = data;
        const originalArticle = responseArticles
          ? responseArticles.find((a: RelatedArticle) => a.id === articleId)
          : null;

        setQueryId(query_id);
        setOriginalArticle(originalArticle);
        setRelatedArticles(responseArticles.filter((a: RelatedArticle) => a.id !== articleId));
        setPage(2);
      } catch {
        setLoading(false);
        setNotFound(true);
        setPage(2);
      }
    };

    fetchData();
  }, [articleId]);

  const loadMoreResults = async () => {
    let response = await fetch(
      `${API_BASE}${RELATED_ENDPOINT}/${articleId.toLowerCase()}?page_number=${page}`,
    );
    setPage(page + 1);

    if (response.status > 400) {
      setHasMore(false);
    }

    let data = await response.json();
    const { response: responseArticles } = data;
    const currentArticles = relatedArticles || [];
    setRelatedArticles([...currentArticles, ...responseArticles]);
  };

  const TitleRow = (
    <Row>
      <RelatedTitle>
        Related Articles <FileText size={24} style={{ marginLeft: '8px' }} />
      </RelatedTitle>
      <SearchLink to={HOME_ROUTE}>
        <ArrowLeft size={16} style={{ marginRight: '4px' }} />
        Search All Articles
      </SearchLink>
    </Row>
  );

  return (
    <PageWrapper>
      <PageContent>
        <ErrorBoundary FallbackComponent={NotFoundComponent}>
          <RelatedContent>
            {loading && <Loading />}
            {notFound && (
              <>
                {TitleRow}
                <NotFoundComponent />
              </>
            )}
            {originalArticle && relatedArticles && (
              <>
                {TitleRow}
                <OriginalArticle>
                  <SmallTitle>Showing articles related to:</SmallTitle>
                  <BaseArticleResult article={originalArticle} boldTitle />
                  {originalArticle.abstract && (
                    <>
                      <AbstractTitle className="hideCollapsed">Abstract</AbstractTitle>
                      <Paragraph>{parseAbstract(originalArticle.abstract)}</Paragraph>
                    </>
                  )}
                </OriginalArticle>
                <InfiniteScroll
                  pageStart={page}
                  loadMore={loadMoreResults}
                  hasMore={hasMore}
                  loader={
                    <Row>
                      <Loading />
                    </Row>
                  }
                >
                  {relatedArticles.map((article, idx) => (
                    <RelatedResult
                      key={article.id}
                      article={article}
                      position={idx}
                      queryId={queryId}
                    />
                  ))}
                </InfiniteScroll>
                {relatedArticles.length === 0 && <NotFound>No related articles found</NotFound>}
              </>
            )}
          </RelatedContent>
        </ErrorBoundary>
      </PageContent>
    </PageWrapper>
  );
};

export default RelatedPage;

const RelatedContent = styled.div`
  width: 100%;
  margin-right: auto;
  display: flex;
  flex-direction: column;
`;

const RelatedTitle = styled.div`
  ${Heading2}
  font-weight: 700;
  font-size: 24px;
  display: flex;
  align-items: center;
`;

const NotFound = styled.div`
  ${Heading3}
  display: flex;
  margin-top: 32px;
  padding-bottom: 24px;
  color: ${({ theme }) => theme.darkGrey};
`;

const SmallTitle = styled.div`
  ${Body}
  padding-bottom: 16px;
  font-weight: 500;
  color: ${({ theme }) => theme.darkGrey};
`;

const Paragraph = styled.div`
  ${BodySmall}
  margin-bottom: 8px;
`;

const AbstractTitle = styled.div`
  ${BodySmall}
  font-weight: 700;
`;

const OriginalArticle = styled.div`
  margin-bottom: 16px;
  border: 1px solid ${({ theme }) => theme.yellow};
  border-bottom: 1px solid ${({ theme }) => theme.yellow};
  padding: 12px 12px 4px 12px;
  border-radius: 4px;
`;

const Row = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 16px;
  padding-bottom: 16px;
  border-bottom: 1px dotted ${({ theme }) => theme.lightGrey};
`;

const SearchLink = styled(RouterLink)`
  ${LinkStyle}
  display: flex;
  align-items: center;
`;
