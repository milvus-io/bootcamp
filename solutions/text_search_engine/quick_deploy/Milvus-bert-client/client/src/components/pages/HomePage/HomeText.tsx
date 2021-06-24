import React from 'react';
import styled from 'styled-components';
import { Body, Link } from '../../../shared/Styles';

const HomeText = () => {
  return (
    <HomeTextWrapper>
      <Paragraph>
        Neural Covidex applies state-of-the-art neural network models and artificial intelligence
        (AI) techniques to answer questions using the{' '}
        <Link
          href="https://pages.semanticscholar.org/coronavirus-research"
          target="_blank"
          rel="noopener noreferrer"
        >
          COVID-19 Open Research Dataset (CORD-19)
        </Link>{' '}
        provided by the{' '}
        <Link href="https://allenai.org/" target="_blank" rel="noopener noreferrer">
          Allen Institute for AI
        </Link>{' '}
        , which currently contains over 140,000 scholarly articles, including over 70,000 with full
        text, about COVID-19 and coronavirus-related research, drawn from a variety of sources
        including PubMed, a curated list of articles from the WHO.
      </Paragraph>

      <Paragraph>
        The{' '}
        <Link
          href="https://github.com/milvus-io/bootcamp/tree/master/EN_solutions"
          target="_blank"
          rel="noopener noreferrer"
        >
          Local Deployment of COVID-19 Dataset Search
        </Link>{' '}
        introduce how to run it, which is referenced from{' '}
        <Link href="https://github.com/castorini/covidex" target="_blank" rel="noopener noreferrer">
          covidex
        </Link>
        , and <Bold>Milvus</Bold> is used to get the related articles. Let's start to have fun with
        it.
      </Paragraph>
    </HomeTextWrapper>
  );
};

export default HomeText;

const HomeTextWrapper = styled.div`
  margin-top: 16px;
`;

const Paragraph = styled.div`
  ${Body}
  margin-bottom: 24px;
`;

const Bold = styled.span`
  ${Body}
  font-weight: 600;
`;
