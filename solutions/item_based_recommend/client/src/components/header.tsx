import React from 'react';
import styled from 'styled-components';

const HeaderWrapper = styled.header`
  display: flex;
  align-items: center;
`;

const HeaderTitle = styled.h2`
  margin-right: 16px;
`;

const Header = () => {
  return (
    <HeaderWrapper>
      <HeaderTitle>Paper Recommend</HeaderTitle>
      <p>powered by Milvus</p>
    </HeaderWrapper>
  );
};

export default Header;
