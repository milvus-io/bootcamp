import React, { useState, useEffect, useCallback } from "react";
import { Search } from "react-feather";
import {
  PageWrapper,
  PageContent,
  Heading1,
  Heading2,
} from "../../../shared/Styles";
import SearchBar from "./SearchBar";
import ErrorBoundary from "react-error-boundary";
import styled from "styled-components";
import Loading from "../../common/Loading";
import {
  SEARCH_API_BASE,
  SEARCH,
  TABLET_BREAKPOINT,
} from "../../../shared/Constants";
import { SearchResultView } from "../../../shared/Models";
import SearchResult from "./SearchResult";
import Navbar from "../../navigation/Navbar";
import Keycodes from "../../../shared/Keycodes";
import MilvusLogo from "../../../img/logo.svg";

const TABLE_NAME = "searchTable";

const SearchPage = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [searchResults, setSearchResults] = useState<SearchResultView[]>([]);
  const [searched, setSearched] = useState<boolean>(false);
  const [inputFocused, setInputFocused] = useState<boolean>(false);
  const [query, setQuery] = useState<string>("");

  useEffect(() => {
    document.title = "Search Engine";
  }, []);

  const handleUserKeyPress = useCallback(
    (event: KeyboardEvent) => {
      if (event.keyCode === Keycodes.ENTER && inputFocused) {
        onSearch(query);
      }
    },
    // eslint-disable-next-line
    [query, inputFocused]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleUserKeyPress);
    return () => {
      window.removeEventListener("keydown", handleUserKeyPress);
    };
  }, [handleUserKeyPress]);

  const onSearch = (query: string) => {
    if (query) {
      fetchSearchResults(query);
      setQuery(query);
    } else {
      if (searched) {
        setSearched(false);
        return;
      }
    }

    setSearched(true);
    setSearchResults([]);
  };

  const fetchSearchResults = async (query: string) => {
    try {
      setLoading(true);
      const response = await fetch(
        `${SEARCH_API_BASE}${SEARCH}?table_name=${TABLE_NAME}&query_sentence=${query}`,
        {
          method: "GET",
          headers: {
            "content-type": "application/json",
          },
        }
      );
      setLoading(false);

      const data = await response.json();
      const searchResults = data.response.map((item: string[]) =>
        formatResData(item)
      );
      setSearchResults(searchResults);
    } catch (err) {
      console.log(err);
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

  const handleInput = (event: React.ChangeEvent<HTMLInputElement>) =>
    setQuery(event.target.value);

  const clearSearchResult = () => {
    setSearchResults([]);
    setSearched(false);
  };

  return (
    <>
      {searchResults.length === 0 && !searched ? (
        <Navbar />
      ) : (
        <NavbarWrapper>
          <PageContent>
            <Row>
              <NavbarLogo tabIndex={0} onClick={clearSearchResult}>
                Search Engine
                <NavbarSubtitle>powered by Milvus</NavbarSubtitle>
              </NavbarLogo>
              <SearchInputWrapper>
                <SearchIcon />
                <SearchBarInput
                  value={query}
                  onChange={handleInput}
                  onFocus={() => setInputFocused(true)}
                  onBlur={() => setInputFocused(false)}
                />
              </SearchInputWrapper>
            </Row>
          </PageContent>
        </NavbarWrapper>
      )}
      <PageWrapper>
        <PageContent>
          {searchResults.length > 0 || searched ? null : (
            <SearchContainer>
              <Logo src={MilvusLogo} alt="logo" />
              <SearchBar
                onSearch={onSearch}
                tableName={TABLE_NAME}
                setLoading={setLoading}
              />
            </SearchContainer>
          )}
          <ErrorBoundary
            FallbackComponent={() => <NoResult>No results found</NoResult>}
          >
            {loading && <Loading />}
            {searchResults.length > 0 ? (
              <SearchResults>
                {searchResults.map((result, i) => {
                  return (
                    <SearchResult
                      result={result}
                      key={`${i}${new Date().toTimeString()}`}
                    />
                  );
                })}
              </SearchResults>
            ) : (
              <NoResult>
                {searched && !loading ? "No results found" : ""}
              </NoResult>
            )}
          </ErrorBoundary>
        </PageContent>
      </PageWrapper>
    </>
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

const NavbarWrapper = styled.div`
  padding: 24px 48px;
  display: flex;
  justify-content: space-between;
  position: relative;
  background-color: #4eb8f0;

  @media only screen and (max-width: ${TABLET_BREAKPOINT}px) {
    padding: 24px 16px;
  }
`;

const Row = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-around;
`;

const NavbarLogo = styled.a`
  display: flex;
  ${Heading1}
  align-items: center;
  position: relative;
  font-weight: 800;
  cursor: pointer;
  color: ${({ theme }) => theme.white};
  max-width: fit-content;
  margin-right: 16px;
`;

const NavbarSubtitle = styled.span`
  display: inline-block;
  margin-left: 8px;
  font-size: 16px;
`;

const SearchInputWrapper = styled.div`
  display: flex;
  flex: 1;
  position: relative;
  align-items: center;
`;

const SearchContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  margin-top: 150px;
`;

const SearchIcon = styled(Search)`
  display: inline;
  height: 16px;
  width: 16px;
  position: absolute;
  top: 10px;
  left: 8px;

  margin-right: 8px;
  color: #fff;
`;

const SearchBarInput = styled.input`
  display: flex;
  width: 60%;
  padding: 8px 12px 8px 32px;
  outline: none;
  color: #fff;
  background-color: transparent;
  border-radius: 4px;
  border: 1px solid transparent;
  background-color: rgba(255, 255, 255, 0.5);

  &:focus {
    border: 1px solid transparent;
  }
`;

const Logo = styled.img`
  width: 40%;
  height: auto;
`;
