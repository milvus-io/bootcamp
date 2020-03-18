import React from "react";
import { makeStyles } from "@material-ui/core/styles";
import QueryProvider from "./contexts/QueryContext";
import Header from "./components/Header";
import QA from "./components/QA";
// import Setting from "./containers/Setting";
import useMediaQuery from "@material-ui/core/useMediaQuery";

const App: React.FC = () => {
  const isMobile = !useMediaQuery("(min-width:1000px)");

  const useStyles = makeStyles({
    root: {
      flexGrow: 1,
      background: "#666769",
      display: isMobile ? "block" : "flex",
      overflow: isMobile ? "auto" : "hidden"
    },
    main: {
      height: "calc(100vh - 60px)",
      maxWidth: "700px",
      margin: "60px auto 0 auto",
      backgroundColor: "#28292E"
    }
  });
  const classes = useStyles({});
  return (
    <QueryProvider>
      <div className={classes.root}>
        <Header></Header>
        {/* {!isMobile && <Setting setImages={setImages} setLoading={setLoading} />} */}
        <div className={classes.main}>
          <QA isMobile={isMobile}></QA>
        </div>
      </div>
    </QueryProvider>
  );
};

export default App;
