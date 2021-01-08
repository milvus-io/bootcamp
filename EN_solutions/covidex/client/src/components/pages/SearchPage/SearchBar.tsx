import React, { useCallback, useState, useEffect } from 'react';
import { Search } from 'react-feather';
import styled from 'styled-components';
import { withRouter, RouteComponentProps } from 'react-router';
import { Button } from 'reakit';

import { LARGE_MOBILE_BREAKPOINT } from '../../../shared/Constants';

import Keycodes from '../../../shared/Keycodes';

interface SearchBarProps extends RouteComponentProps {
  onSearch: (query: string) => any;
}

const SearchBar = ({ onSearch }: SearchBarProps) => {
  const [inputFocused, setInputFocused] = useState<boolean>(false);
  const [query, setQuery] = useState<string>('');

  const handleInput = (event: React.ChangeEvent<HTMLInputElement>) => setQuery(event.target.value);

  const submitQuery = (q: string = query) => {
    onSearch(q);
  };

  const handleUserKeyPress = useCallback(
    (event: KeyboardEvent) => {
      if (event.keyCode === Keycodes.ENTER && inputFocused) {
        submitQuery();
      }
    },
    // eslint-disable-next-line
    [query, inputFocused],
  );

  useEffect(() => {
    window.addEventListener('keydown', handleUserKeyPress);
    return () => {
      window.removeEventListener('keydown', handleUserKeyPress);
    };
  }, [handleUserKeyPress]);

  return (
    <SearchBarWrapper>
      <Section>
        <SearchInputWrapper>
          <SearchBarInput
            value={query}
            onChange={handleInput}
            onSubmit={() => submitQuery()}
            onFocus={() => setInputFocused(true)}
            onBlur={() => setInputFocused(false)}
          />
        </SearchInputWrapper>
        <SearchButton
          type="submit"
          onSubmit={() => submitQuery()}
          onClick={() => submitQuery()}
          onMouseDown={(e: any) => e.preventDefault()}
        >
          <SearchIcon />
          Search
        </SearchButton>
      </Section>
    </SearchBarWrapper>
  );
};

export default withRouter(SearchBar);

const SearchBarWrapper = styled.div`
  position: relative;
  margin-right: auto;
  display: flex;
  margin-bottom: 24px;
  width: 860px;
  max-width: 100%;

  @media only screen and (max-width: ${LARGE_MOBILE_BREAKPOINT}px) {
    flex-direction: column;
  }
`;

const SearchIcon = styled(Search)`
  display: inline;
  height: 16px;
  width: 16px;
  margin-right: 8px;
`;

const SearchBarInput = styled.input`
  display: flex;
  width: 100%;
  padding: 12px 16px;
  outline: none;
  border-radius: 4px 0 0 4px;
  border: 1px solid ${({ theme }) => theme.grey};
  border-right: none;

  &:focus {
    border: 1px solid ${({ theme }) => theme.primary};
    border-right: none;
  }
`;

const SearchButton = styled(Button)`
  display: flex;
  background: ${({ theme }) => theme.primary};
  border: none;
  padding: 12px 16px;
  cursor: pointer;
  color: ${({ theme }) => theme.white};
  border-radius: 0 4px 4px 0;
  outline: none;
  transition: background 0.1s;

  &:hover,
  &:focus {
    background: ${({ theme }) => theme.secondary};
  }
`;

const SearchInputWrapper = styled.div`
  display: flex;
  flex: 1;
  position: relative;
`;

const Section = styled.div`
  display: flex;

  &:last-child {
    flex: 1;
  }

  @media only screen and (max-width: ${LARGE_MOBILE_BREAKPOINT}px) {
    &:last-child {
      margin-top: 16px;
    }
  }
`;
