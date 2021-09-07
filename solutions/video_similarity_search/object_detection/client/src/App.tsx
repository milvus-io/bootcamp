import React from "react";
import logo from "./logo.svg";
import "./App.css";
import RootPrvider from "./context/rootProvider";
import HomePage from "./pages/home";

function App() {
  return (
    <div className="App">
      <RootPrvider>
        <HomePage />
      </RootPrvider>
    </div>
  );
}

export default App;
