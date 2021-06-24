import React from 'react';
import styled from 'styled-components';
import { withRouter, RouteComponentProps } from 'react-router';

import { TABLET_BREAKPOINT, HOME_ROUTE } from '../../shared/Constants';
import { PageContent, Heading1 } from '../../shared/Styles';

const Navbar = ({ history }: RouteComponentProps) => {
  const isSearch = history.location.pathname === '/search';
  const handleClick = () => {
    if (!isSearch) {
      history.push(HOME_ROUTE);
    }
  };
  return (
    <NavbarWrapper>
      <PageContent>
        <Row>
          <NavbarLogo tabIndex={0} onClick={handleClick}>
            {isSearch ? 'Search Engine' : 'Neural Covidex'}
            <NavbarSubtitle>powered by Milvus</NavbarSubtitle>
          </NavbarLogo>
        </Row>
      </PageContent>
    </NavbarWrapper>
  );
};

export default withRouter(Navbar);

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
  justify-content: space-between;
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
  margin-right: 8px;
`;

const NavbarSubtitle = styled.span`
  display: inline-block;
  margin-left: 8px;
  font-size: 16px;
`;
