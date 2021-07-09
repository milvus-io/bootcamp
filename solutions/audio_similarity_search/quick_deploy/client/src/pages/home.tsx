import React, { useState, useContext, useEffect } from "react";
import Header from "../components/header";
import SideBar from "../components/sidebar";
import Uploader from "../components/uploader";
import Search from "../components/search";
import { makeStyles, Theme } from "@material-ui/core";
import { rootContext } from "../context";

const useStyles = makeStyles((theme: Theme) => ({
  main: {
    display: "flex",
    height: "100%",
  },
  content: {
    width: "100%",
    height: "100%",
  },
}));

const Home = () => {
  const { count: getCount, tableName } = useContext(rootContext);
  const classes = useStyles();
  const [count, setCount] = useState(0);
  const [tab, setTab] = useState<"upload" | "search">("upload");

  const getCountNum = async () => {
    const res = await getCount({ table_name: tableName });
    console.log(res);
  };

  useEffect(() => {
    getCountNum();
  }, []);
  return (
    <>
      <Header />
      <div className={classes.main}>
        <SideBar setTab={setTab} count={count} tab={tab} />
        <div className={classes.content}>
          {tab === "upload" ? <Uploader setTab={setTab} /> : <Search />}
        </div>
      </div>
    </>
  );
};

export default Home;
