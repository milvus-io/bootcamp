import React, { useState } from "react";
import { makeStyles } from "@material-ui/core/styles";
import useMediaQuery from "@material-ui/core/useMediaQuery";
import Setting from "../containers/Setting";
import Results from "./Results";
const SearchPage: React.FC = () => {
  const isMobile = !useMediaQuery("(min-width:1000px)");
  const classes = makeStyles({
    root: {
      flexGrow: 1,
      display: isMobile ? 'block' : 'flex'
    }
  })()
  const [results, setResults]: any = useState([]);
  return (
    <div className={classes.root}>
      <Setting results={results} setResults={setResults} />
      <Results results={results} setResults={setResults} />
    </div>
  );
};

export default SearchPage;
