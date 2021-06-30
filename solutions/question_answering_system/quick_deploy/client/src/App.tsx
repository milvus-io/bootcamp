import React, { useState } from "react";
import { makeStyles } from "@material-ui/core/styles";
import QueryProvider from "./contexts/QueryContext";
import Header from "./components/Header";
import Menu from "./components/Menu";

import QA from "./components/QA";
// import Setting from "./containers/Setting";
import useMediaQuery from "@material-ui/core/useMediaQuery";

const App: React.FC = () => {
  const isMobile = !useMediaQuery("(min-width:768px)");
  const [loading, setLoading] = useState(false);
  const useStyles = makeStyles({
    root: {
      flexGrow: 1,
      display: isMobile ? "block" : "flex",
      overflow: isMobile ? "auto" : "hidden",
      maxWidth: "700px",
      margin: "4rem auto",
      boxShadow:
        " 0 1.1px 2.9px rgba(0, 0, 0, 0.044),0 2.5px 6.8px rgba(0, 0, 0, 0.063),0 4.5px 12.3px rgba(0, 0, 0, 0.078),0 7.4px 20.4px rgba(0, 0, 0, 0.092),0 12.2px 33.6px rgba(0, 0, 0, 0.107),0 21.3px 58.7px rgba(0, 0, 0, 0.126),0 46px 127px rgba(0, 0, 0, 0.17)",
    },
    menu: {
      flex: "0 0 100px",
    },
    main: {
      position: "relative",
      display: "flex",
      flexDirection: "column",
      backgroundColor: "#fff",
      height: "100%",
      width: "100%",
    },
  });
  const classes = useStyles({});
  return (
    <QueryProvider>
      <div className={classes.root}>
        {!isMobile && <Menu></Menu>}
        {/* {!isMobile && <Setting setImages={setImages} setLoading={setLoading} />} */}
        <div className={classes.main}>
          <Header loading={loading}></Header>
          <QA
            isMobile={isMobile}
            loading={loading}
            setLoading={setLoading}
          ></QA>
        </div>
      </div>
    </QueryProvider>
  );
};

export default App;
