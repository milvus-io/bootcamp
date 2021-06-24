import React from "react";
import RootContainer from './containers/RootContainer'
import {
  BrowserRouter as Router,
  Switch,
  Route,
} from 'react-router-dom'
import './App.css'

const App: React.FC = () => {
  return (
    <Router>
      <Switch>
        <Route exact path="/" component={RootContainer} />
      </Switch>
    </Router>
  );
};

export default App;
