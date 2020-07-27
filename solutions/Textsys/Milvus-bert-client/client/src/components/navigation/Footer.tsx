import React from 'react';
import styled from 'styled-components';

import NYU from '../../img/nyu.png';
import UWaterloo from '../../img/uwaterloo.png';

import { Link, PageContent } from '../../shared/Styles';
import { TABLET_BREAKPOINT } from '../../shared/Constants';

const Footer = () => {
  return (
    <FooterWrapper>
      <PageContent>
        <Images>
          <Link href="https://uwaterloo.ca/" target="_blank" rel="noopener noreferrer">
            <SchoolImage src={UWaterloo} alt="University of Waterloo Logo" marginRight={24} />
          </Link>
          <Link href="https://www.nyu.edu/" target="_blank" rel="noopener noreferrer">
            <SchoolImage src={NYU} alt="NYU Logo" />
          </Link>
        </Images>
      </PageContent>
    </FooterWrapper>
  );
};

export default Footer;

const FooterWrapper = styled.div`
  padding: 24px 48px;
  display: flex;
  position: relative;
  flex-direction: column;

  @media only screen and (max-width: ${TABLET_BREAKPOINT}px) {
    padding: 24px 16px;
  }
`;

const Images = styled.div`
  margin-top: 16px;
  display: flex;
  position: relative;
  justify-content: flex-end;
`;

const SchoolImage = styled.img<{ marginRight?: number }>`
  height: 28px;
  width: auto;
  margin-right: ${({ marginRight }) => marginRight || 0}px;
`;
