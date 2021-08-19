import React from 'react';
import { BrowserRouter as Router, Switch, Route } from 'react-router-dom';
import styled, { ThemeProvider } from 'styled-components';

import Navbar from './components/navigation/Navbar';
import RelatedPage from './components/pages/RelatedPage/RelatedPage';
import NotFoundPage from './components/pages/NotFoundPage';

import { HOME_ROUTE, RELATED_ROUTE } from './shared/Constants';
import Theme from './shared/Theme';
import SearchPage from './components/pages/SearchPage/SearchPage';

const App = () => {
  return (
    <ThemeProvider theme={Theme}>
      <Router>
        <AppContainer>
          <Switch>
            <Route path={HOME_ROUTE}>
              <SearchPage />
            </Route>
            <Route path={`${RELATED_ROUTE}/:articleId`}>
              <Navbar />
              <RelatedPage />
            </Route>
            <Route>
              <Navbar />
              <NotFoundPage />
            </Route>
          </Switch>
        </AppContainer>
      </Router>
    </ThemeProvider>
  );
};

export default App;

const AppContainer = styled.div`
  display: flex;
  flex-direction: column;
  min-height: 100vh;
`;
