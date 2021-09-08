import React from "react";
import "./App.css";
import RootPrvider from "./context/rootProvider";
import Home from "./pages/home";
import Header from "./components/header";

function App() {
  return (
    <div className="App">
      <RootPrvider>
        <Header />
        <Home />
      </RootPrvider>
    </div>
  );
}

export default App;
