import React from 'react';
import ReactDOM from 'react-dom';
import QueryProvider from "./contexts/QueryContext";

import './index.css';
import App from './App';
import * as serviceWorker from './serviceWorker';

ReactDOM.render(
  <QueryProvider>
    <App />
  </QueryProvider>
  , document.getElementById('root'));

serviceWorker.unregister();
