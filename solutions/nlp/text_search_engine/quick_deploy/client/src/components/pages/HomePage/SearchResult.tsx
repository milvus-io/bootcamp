import React, { useState, ReactNode, useRef } from 'react';
import styled, { css } from 'styled-components';
import { ChevronsDown, Link as LinkIcon } from 'react-feather';
import Highlighter from 'react-highlight-words';
import { Link } from 'react-router-dom';

import { LinkStyle, BodySmall, FadeInText } from '../../../shared/Styles';
import {
  API_BASE,
  SEARCH_COLLAPSED_ENDPOINT,
  SEARCH_EXPANDED_ENDPOINT,
  SEARCH_CLICKED_ENDPOINT,
  RELATED_ROUTE,
} from '../../../shared/Constants';
import { makePOSTRequest, parseAbstract } from '../../../shared/Util';
import { SearchArticle } from '../../../shared/Models';
import BaseArticleResult from '../../common/BaseArticleResult';

interface SearchResultProps {
  article: SearchArticle;
  position: number;
  queryId: string;
  queryTokens: Array<string>;
}

const highlightText = (
  text: string,
  highlights: Array<[number, number]>,
  queryTokens: Array<string>,
): Array<string | ReactNode> => {
  if (!highlights) {
    return [
      <TextSpan className="hideCollapsed" key={0}>
        {text}
      </TextSpan>,
    ];
  }

  let highlighted: Array<string | ReactNode> = [];
  let prevEnd = -1;

  highlights.forEach((highlight, i) => {
    const [start, end] = highlight;
    highlighted.push(
      <Ellipsis key={`${i}-ellipsis-1`} className="showCollapsed">
        ...
      </Ellipsis>,
    );
    highlighted.push(
      <TextSpan key={`${i}-1`} className="hideCollapsed">
        {highlightMatches(queryTokens, text.substr(prevEnd + 1, start - prevEnd - 1))}
      </TextSpan>,
    );
    highlighted.push(
      <Highlight className="highlight" key={`${i}-2`}>
        {highlightMatches(queryTokens, text.substr(start, end - start + 1))}
      </Highlight>,
    );

    prevEnd = end;
  });

  // add last part of text
  if (prevEnd < text.length) {
    highlighted.push(
      <TextSpan key="last" className="hideCollapsed">
        {highlightMatches(queryTokens, text.substr(prevEnd + 1, text.length - prevEnd - 1))}
      </TextSpan>,
    );
  }

  return highlighted;
};

const highlightMatches = (queryTokens: Array<string>, text: string): ReactNode => {
  return (
    <Highlighter
      searchWords={queryTokens}
      textToHighlight={text}
      highlightTag={Match}
      highlightClassName="match"
    />
  );
};

// adjust highlights based on difference
const adjustHighlights = (
  highlights: Array<[number, number]>,
  adjustment: number,
): Array<[number, number]> => {
  if (!highlights) {
    return [];
  }

  return highlights.map((highlight) => [highlight[0] + adjustment, highlight[1] + adjustment]);
};

const SearchResult = ({ article, position, queryId, queryTokens }: SearchResultProps) => {
  const fullTextRef = useRef(null);
  const [collapsed, setCollapsed] = useState<boolean>(true);

  // Separate abstract from other paragraphs if it was highlighted
  const originalAbstract = article.highlighted_abstract
    ? article.paragraphs[0]
    : article.abstract || '';

  const abstract = parseAbstract(originalAbstract);
  const abstractHighlights = article.highlighted_abstract
    ? adjustHighlights(article.highlights[0], abstract.length - originalAbstract.length)
    : [];

  const paragraphs = article.highlighted_abstract
    ? article.paragraphs.slice(1)
    : article.paragraphs;
  const highlights = article.highlighted_abstract
    ? article.highlights.slice(1)
    : article.highlights;

  const interactionRequestBody = { query_id: queryId, result_id: article.id, position };

  return (
    <SearchResultWrapper>
      <BaseArticleResult
        article={article}
        position={position}
        onClickTitle={() =>
          makePOSTRequest(`${API_BASE}${SEARCH_CLICKED_ENDPOINT}`, interactionRequestBody)
        }
      />
      <div ref={fullTextRef}>
        {/* Display abstract */}
        {abstract && (
          <>
            <ResultText collapsed={collapsed} marginBottom={4}>
              <SectionTitle className="hideCollapsed">Abstract</SectionTitle>
            </ResultText>
            <Paragraph marginBottom={16} collapsed={collapsed}>
              {highlightText(abstract, abstractHighlights, queryTokens)}
            </Paragraph>
          </>
        )}
        {/* Display paragraphs */}
        {paragraphs && paragraphs.length > 0 && (
          <ResultText collapsed={collapsed} marginTop={20} marginBottom={4}>
            <SectionTitle className="hideCollapsed">Full-Text Excerpt</SectionTitle>
          </ResultText>
        )}
        {paragraphs.map((paragraph, i) => (
          <Paragraph marginTop={i === 0 ? 0 : 16} key={i} collapsed={collapsed}>
            {highlightText(paragraph, highlights[i], queryTokens)}
            {i === paragraphs.length - 1 && highlights[i] && highlights[i].length > 0 && (
              <Ellipsis className="showCollapsed">...</Ellipsis>
            )}
          </Paragraph>
        ))}
      </div>
      <LinkContainer>
        {(abstract || paragraphs.length > 0) && (
          <TextLink
            onClick={() => {
              makePOSTRequest(
                `${API_BASE}${collapsed ? SEARCH_EXPANDED_ENDPOINT : SEARCH_COLLAPSED_ENDPOINT}`,
                interactionRequestBody,
              );
              setCollapsed(!collapsed);
            }}
            onMouseDown={(e) => e.preventDefault()}
          >
            {collapsed ? 'Show more' : 'Show less'}
            <Chevron collapsed={collapsed} />
          </TextLink>
        )}
        {article.has_related_articles && (
          <RelatedLink to={`${RELATED_ROUTE}/${article.id}`}>
            Related articles <LinkIcon size={12} style={{ marginLeft: '4px' }} />
          </RelatedLink>
        )}
      </LinkContainer>
    </SearchResultWrapper>
  );
};

export default SearchResult;

const SearchResultWrapper = styled.div`
  display: flex;
  flex-direction: column;
  width: 100%;
  margin: 0 auto;
  padding: 24px 0;
  border-bottom: 1px dotted ${({ theme }) => theme.lightGrey};
  margin-bottom: 8px;
`;

const fadeInAnimation = css`animation ${FadeInText} 0.5s ease-in-out;`;

const ResultText = styled.div<{
  marginTop?: number;
  marginBottom?: number;
  collapsed: boolean;
}>`
  ${BodySmall}
  color: ${({ theme }) => theme.darkGrey};
  margin-top: ${({ marginTop, collapsed }) => (marginTop && !collapsed ? marginTop : 0)}px;
  margin-bottom: ${({ marginBottom, collapsed }) =>
    marginBottom && !collapsed ? marginBottom : 0}px;
  display: ${({ collapsed }) => (collapsed ? 'inline' : 'block')};

  & > .showCollapsed {
    opacity: ${({ collapsed }) => (collapsed ? 1 : 0)};
    display: ${({ collapsed }) => (collapsed ? 'inline' : 'none')};
    ${({ collapsed }) => (collapsed ? fadeInAnimation : '')}
  }

  & > .hideCollapsed {
    opacity: ${({ collapsed }) => (collapsed ? 0 : 1)};
    display: ${({ collapsed }) => (collapsed ? 'none' : 'inline')};
    ${({ collapsed }) => (collapsed ? '' : fadeInAnimation)}
  }

  & > .highlight {
    background: ${({ theme, collapsed }) => (collapsed ? 'none' : theme.paleYellow)};
  }
`;

const Paragraph = styled(ResultText)`
  ${({ theme, collapsed }) =>
    collapsed
      ? ''
      : `
    padding-left: 8px;
    border-left: 1px solid ${theme.lightGrey};
  `}
`;

const LinkContainer = styled.div`
  display: flex;
  margin-top: 8px;
`;

const TextLink = styled.button`
  ${BodySmall}
  ${LinkStyle}
  max-width: fit-content;
  margin-right: 16px;
  display: flex;
  align-items: center;
  background: none;
  padding: 0;
  border: none;
`;

const RelatedLink = styled(Link)`
  ${BodySmall}
  ${LinkStyle}
  margin-right: 16px;
  display: flex;
  align-items: center;
`;

const Chevron = styled(ChevronsDown)<{ collapsed: boolean }>`
  height: 14px;
  width: 14px;
  transform: rotate(${({ collapsed }) => (collapsed ? 0 : 180)}deg);
  transition: 550ms transform;
`;

const TextSpan = styled.span`
  line-height: 1.4;
  transition: all 0.3s;
`;

const Ellipsis = styled(TextSpan)`
  letter-spacing: 1px;
  margin: 0 2px;
`;

const Highlight = styled(TextSpan)`
  position: relative;
  font-weight: 400;

  & > span > .match {
    font-weight: 600;
  }
`;

const Match = styled(TextSpan)`
  position: relative;
  font-weight: 500;
`;

const SectionTitle = styled.div`
  ${BodySmall}
  color: ${({ theme }) => theme.slate};
  font-weight: 600;
`;
