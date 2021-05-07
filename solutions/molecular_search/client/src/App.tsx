import React, { useState,  } from "react";
import { makeStyles } from "@material-ui/core/styles";
import QueryProvider from "./contexts/QueryContext";
import Setting from "./containers/Setting";
import SearchResults from "./components/ChemistryResults";

const useStyles = makeStyles({
  root: {
    flexGrow: 1,
    display: "flex",
    overflow: "hidden"
  }
});

const App: React.FC = () => {
  const classes = useStyles({});
  const [results, setResults]: any = useState([]);
  const [loading, setLoading]: any = useState(false);
  return (
    <QueryProvider>
      <div className={classes.root}>
        <Setting setResults={setResults} setLoading={setLoading} />
        <SearchResults results={results} />
        {loading && (
          <div
            style={{
              position: "absolute",
              width: "100%",
              height: "100%",
              left: "0",
              top: "0",
              backgroundColor: "#000",
              opacity: 0.5
            }}
          ></div>
        )}
      </div>
    </QueryProvider>
  );
};

export default App;
