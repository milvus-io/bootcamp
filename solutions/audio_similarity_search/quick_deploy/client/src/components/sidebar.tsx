import React from "react";
import { makeStyles, Theme } from "@material-ui/core";
import BackupIcon from "@material-ui/icons/Backup";
import SearchIcon from "@material-ui/icons/Search";
import IconButton from "@material-ui/core/IconButton";

const useStyles = makeStyles((theme: Theme) => ({
  sidebarWrapper: {
    height: "calc(100vh - 64px)",
    width: "80px",
    background: "#fff",
  },
  iconWrapper: {
    width: "80px",
    height: "80px",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    cursor: "pointer",
    borderRadius: 0,

    "&:hover": {
      background: "#F3FCFE",
      color: "#12c3f4",
    },
    "& svg": {
      fontSize: "40px",
    },
  },
  avtice: {
    background: "#12c3f4",
    color: "#fff",
  },
}));

type PropsType = {
  setTab: React.Dispatch<React.SetStateAction<"upload" | "search">>;
  count: number;
  tab: "upload" | "search";
};

const SideBar: React.FC<PropsType> = ({ setTab, tab, count }) => {
  const classes = useStyles();
  return (
    <div className={classes.sidebarWrapper}>
      <IconButton
        aria-label="upload"
        classes={{ root: classes.iconWrapper }}
        className={`${tab === "upload" ? classes.avtice : ""}`}
        onClick={() => setTab("upload")}
      >
        <BackupIcon />
      </IconButton>

      <IconButton
        aria-label="search"
        disabled={count === 0}
        classes={{ root: classes.iconWrapper }}
        className={`${tab === "search" ? classes.avtice : ""}`}
        onClick={() => setTab("search")}
      >
        <SearchIcon />
      </IconButton>
    </div>
  );
};

export default SideBar;
