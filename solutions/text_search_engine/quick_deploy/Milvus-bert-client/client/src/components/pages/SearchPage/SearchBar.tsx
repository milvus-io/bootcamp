import React, { useCallback, useState, useEffect, useRef } from "react";
import { Search, Plus } from "react-feather";
import styled from "styled-components";
import { withRouter, RouteComponentProps } from "react-router";
import { Button } from "reakit";

import {
  SEARCH_API_BASE,
  LARGE_MOBILE_BREAKPOINT,
  DROP,
  LOAD,
} from "../../../shared/Constants";
import Keycodes from "../../../shared/Keycodes";

interface SearchBarProps extends RouteComponentProps {
  onSearch: (query: string) => any;
  tableName: string;
  setLoading: React.Dispatch<React.SetStateAction<boolean>>;
}

const SearchBar = ({ onSearch, tableName, setLoading }: SearchBarProps) => {
  const [inputFocused, setInputFocused] = useState<boolean>(false);
  const [query, setQuery] = useState<string>("");
  const inputRef = useRef<HTMLInputElement>(null!);

  const handleInput = (event: React.ChangeEvent<HTMLInputElement>) =>
    setQuery(event.target.value);

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
    [query, inputFocused]
  );

  const deleteTable = async () => {
    const res = await fetch(`${SEARCH_API_BASE}${DROP}`, {
      method: "POST",
      body: JSON.stringify({ table_name: tableName }),
      headers: {
        "content-type": "application/json",
      },
    });

    try {
      const data = await res.json();
      return data;
    } catch (error) {
      console.log(error);
    }
  };

  const handleUploadTable = async () => {
    setLoading(true);
    // delete table before upload
    try {
      await deleteTable();
    } catch (error) {
      setLoading(false);
    }

    const fd = new FormData();
    const file = inputRef.current.files![0];
    if (file) {
      const fileType = file.name.split(".")[1];
      if (fileType !== "csv") {
        alert("type error");
        return;
      }
      fd.append("file", file);
      fd.append("table_name", tableName);

      try {
        const res = await fetch(`${SEARCH_API_BASE}${LOAD}`, {
          method: "POST",
          body: fd,
        });
        const data = res.json();
        console.log(data);
      } catch (error) {
        console.log(error);
      } finally {
        setLoading(false);
      }
    }
  };

  useEffect(() => {
    window.addEventListener("keydown", handleUserKeyPress);
    return () => {
      window.removeEventListener("keydown", handleUserKeyPress);
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
        <UploadFileBtn>
          <UploadIcon />
          Upload
          <FileUploader
            ref={inputRef}
            type="file"
            onChange={handleUploadTable}
          />
        </UploadFileBtn>
      </Section>
    </SearchBarWrapper>
  );
};

export default withRouter(SearchBar);

const SearchBarWrapper = styled.div`
  position: relative;
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

const UploadIcon = styled(Plus)`
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
    border: 1px solid #4eb8f0;
    border-right: none;
  }
`;

const SearchButton = styled(Button)`
  display: flex;
  background: #4eb8f0;
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

const UploadFileBtn = styled(Button)`
  position: relative;
  display: flex;
  align-items: center;
  background: #4eb8f0;
  border: 1px solid #99d3f5;
  border-radius: 4px;
  padding: 4px 12px;
  overflow: hidden;
  color: ${({ theme }) => theme.white};
  text-indent: 0;
  line-height: 20px;

  &:hover {
    text-decoration: none;
    background: ${({ theme }) => theme.secondary};
  }
`;

const FileUploader = styled.input`
  position: absolute;
  width: 100%;
  height: 100%;
  right: 0;
  top: 0;
  opacity: 0;
  visibility: hiiden;
`;
