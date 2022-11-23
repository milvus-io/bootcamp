import React from "react";
import Home from "./pages/home";
import "./App.css";
import ContextProvider from "./context";

function App() {
  return (
    <div className="App">
      <ContextProvider>
        <Home />
      </ContextProvider>
    </div>
  );
}

export default App;
