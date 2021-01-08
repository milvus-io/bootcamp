import React, { useState } from 'react';
import { PageWrapper, PageContent, Heading2 } from '../../../shared/Styles';
import SearchBar from './SearchBar';
import ErrorBoundary from 'react-error-boundary';
import styled from 'styled-components';
import Loading from '../../common/Loading';
import { SEARCH_API_BASE, SEARCH_PAGE_ENDPOINT } from '../../../shared/Constants';
import { SearchResultView } from '../../../shared/Models';
import SearchResult from './SearchResult';

const SearchPage = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [searchResults, setSearchResults] = useState<SearchResultView[]>([]);
  const [searched, setSearched] = useState<boolean>(false);

  const onSearch = (query: string) => {
    if (query) {
      fetchSearchResults(query);
    }

    setSearched(true);
  };

  const fetchSearchResults = async (query: string) => {
    try {
      setLoading(true);

      const response = await fetch(`${SEARCH_API_BASE}${SEARCH_PAGE_ENDPOINT}`, {
        method: 'POST',
        body: JSON.stringify({ query_text: query }),
        headers: {
          'content-type': 'application/json',
        },
      });
      setLoading(false);

      const data = await response.json();
      const searchResults = data.response.map((item: string[]) => formatResData(item));
      setSearchResults(searchResults);
    } catch (err) {
      setLoading(false);
    }
  };

  const formatResData = (data: string[]): SearchResultView => {
    const [title, content] = data;

    return {
      title,
      content,
    } as SearchResultView;
  };

  return (
    <PageWrapper>
      <PageContent>
        <SearchBar onSearch={onSearch} />

        <ErrorBoundary FallbackComponent={() => <NoResult>No results found</NoResult>}>
          {loading && <Loading />}
          {searchResults.length > 0 ? (
            <SearchResults>
              {searchResults.map((result, i) => {
                return <SearchResult result={result} key={`${i}${new Date().toTimeString()}`} />;
              })}
            </SearchResults>
          ) : (
            <NoResult>{searched && !loading ? 'No results found' : ''}</NoResult>
          )}
        </ErrorBoundary>
      </PageContent>
    </PageWrapper>
  );
};

export default SearchPage;

const NoResult = styled.div`
  ${Heading2}
  display: flex;
  margin-top: 16px;
  padding-bottom: 24px;
`;

const SearchResults = styled.div`
  display: flex;
  flex-direction: column;
`;
