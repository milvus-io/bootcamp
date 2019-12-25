import React, { useState } from "react";
import { createStyles, Theme, makeStyles } from "@material-ui/core/styles";
import QueryProvider from "./contexts/QueryContext";
import Setting from "./containers/Setting";
import SearchResults from "./components/SearchResults";

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    root: {
      flexGrow: 1,
      display: "flex",
      overflow: "hidden"
    }
  })
);

const App: React.FC = () => {
  const classes = useStyles({});
  const [images, setImages]: any = useState([]);
  const [loading, setLoading]: any = useState(false);
  return (
    <QueryProvider>
      <div className={classes.root}>
        <Setting setImages={setImages} setLoading={setLoading} />
        <SearchResults images={images} />
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
