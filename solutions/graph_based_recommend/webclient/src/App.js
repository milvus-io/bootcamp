import React from 'react';
import { BrowserRouter as Router, Switch, Route } from 'react-router-dom';
import './App.css';
import Footer from './components/Footer';
import DetailPage from './pages/detail';
import HomePage from './pages/home';
import RecommendPage from './pages/recommend';
import { DETAIL_ROUTE, HOME_ROUTE, RECOMMEND_ROUTE } from './shared/constants';

function App() {
  return (
    <Router>
      <div className="app-wrapper">
        <div className="app-content">
          <Switch>
            <Route exact path={HOME_ROUTE}>
              <HomePage />
            </Route>
            <Route path={RECOMMEND_ROUTE}>
              <RecommendPage />
            </Route>
            <Route path={DETAIL_ROUTE}>
              <DetailPage />
            </Route>
          </Switch>
          <Footer />
        </div>
      </div>
    </Router>
  );
}

export default App;
