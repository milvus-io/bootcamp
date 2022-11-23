import React, { useState } from 'react';
import styled, { css } from 'styled-components';
import { ChevronsDown } from 'react-feather';

import { LinkStyle, BodySmall, FadeInText, Heading4 } from '../../../shared/Styles';

import { SearchResultView } from '../../../shared/Models';

interface SearchResultProps {
  result: SearchResultView;
}

const SearchResult = ({ result }: SearchResultProps) => {
  const [collapsed, setCollapsed] = useState<boolean>(true);
  return (
    <SearchResultWrapper>
      <Title>{result.title}</Title>
      <ResultText collapsed={collapsed} marginTop={20} marginBottom={4}>
        <Paragraph marginBottom={16} collapsed={collapsed}>
          {result.content}
        </Paragraph>
      </ResultText>
      <LinkContainer>
        {result.content && (
          <TextLink
            onClick={() => {
              setCollapsed(!collapsed);
            }}
            onMouseDown={(e) => e.preventDefault()}
          >
            {collapsed ? 'Show more' : 'Show less'}
            <Chevron collapsed={collapsed} />
          </TextLink>
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

const Title = styled.div<{ bold?: boolean }>`
  ${Heading4}
  margin-bottom: 16px;
  font-weight: ${({ bold }) => (bold ? 700 : 400)};
`;

const Paragraph = styled(ResultText)`
  ${({ theme, collapsed }) =>
    collapsed
      ? `
        -webkit-line-clamp: 3;
        display: -webkit-box;
        font-size: 20px;
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-overflow: ellipsis;
      `
      : `
        font-size: 20px;
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

const Chevron = styled(ChevronsDown)<{ collapsed: boolean }>`
  height: 14px;
  width: 14px;
  transform: rotate(${({ collapsed }) => (collapsed ? 0 : 180)}deg);
  transition: 550ms transform;
`;
