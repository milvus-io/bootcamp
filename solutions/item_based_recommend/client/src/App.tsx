import React from 'react';
import { ThemeProvider } from 'styled-components';
import { BrowserRouter as Router, Switch, Route } from 'react-router-dom';
import HomePage from './pages/home/home';
import RecommendPage from './pages/recommend/recommend';
import { HOME_ROUTE, RECOMMEND_ROUTE } from './shared/constansts';
import theme from './shared/theme';

function App() {
  return (
    <ThemeProvider theme={theme}>
      <Router>
        <Switch>
          <Route exact path={HOME_ROUTE}>
            <HomePage />
          </Route>
          <Route path={RECOMMEND_ROUTE}>
            <RecommendPage />
          </Route>
        </Switch>
      </Router>
    </ThemeProvider>
  );
}

export default App;
