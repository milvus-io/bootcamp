import React from "react";
import {
  IconButton,
  makeStyles,
  createStyles,
  Tooltip,
} from "@material-ui/core";
import { Search, CloudUpload } from "@material-ui/icons";
import HelpIcon from "@material-ui/icons/Help";
import FaceIcon from "@material-ui/icons/Face";
import NotListedLocationIcon from "@material-ui/icons/NotListedLocation";

const useStyles = makeStyles((theme) =>
  createStyles({
    root: {
      position: "absolute",
      top: 0,
      left: 0,
      bottom: 0,
      width: "60px",
      background: "#fff",
      display: "flex",
      flexDirection: "column",
      justifyContent: "space-between",
      alignItems: "center",
    },
    btn: {
      width: "100%",
      height: "60px",
      borderRadius: 0,
      "&:hover": {
        backgroundColor: "#4fc4f9",
        color: "#fff",
        svg: {
          color: "#fff",
        },
      },
    },
    search: {
      backgroundColor: "#F3FCFE",
      color: "#12C3F4",
    },
    input: {
      display: "none",
    },
    help: {
      marginBottom: theme.spacing(2),
    },
    active: {
      backgroundColor: "#4fc4f9",
      color: "#fff",
      svg: {
        color: "#fff",
      },
    },
  })
);

const DemoMenu = (props: any) => {
  const classes = useStyles();
  const [active, setActive] = React.useState("");
  const {
    handleUploadData,
    searchDisable,
    inputChangeCb,
    inputRef,
    handleSearchInput,
    acceptFileTypes,
    handleFaceAnalysis,
  } = props;

  const handleInputChange = async (e: any) => {
    const file = inputRef.current.files[0] || "";
    if (!file) {
      return;
    }

    inputChangeCb(file);
  };
  return (
    <div className={classes.root}>
      <div>
        <IconButton
          className={`${classes.btn} ${classes.search}`}
          onClick={handleUploadData}
        >
          <CloudUpload />
        </IconButton>
        <IconButton
          className={`${classes.btn} ${active === "search" && classes.active}`}
          onClick={() => {
            setActive("search");
            handleSearchInput();
          }}
          disabled={searchDisable}
        >
          {handleFaceAnalysis ? <NotListedLocationIcon /> : <Search />}
        </IconButton>
        {handleFaceAnalysis && (
          <IconButton
            className={`${classes.btn} ${active === "face" && classes.active}`}
            onClick={() => {
              setActive("face");
              handleFaceAnalysis();
            }}
            disabled={searchDisable}
          >
            <FaceIcon></FaceIcon>
          </IconButton>
        )}
      </div>

      <div className={classes.help}>
        <Tooltip placement="top" title="Email USO@zilliz.com for support">
          <HelpIcon />
        </Tooltip>
      </div>

      <input
        type="file"
        ref={inputRef}
        className={classes.input}
        onChange={handleInputChange}
        accept={acceptFileTypes}
      ></input>
    </div>
  );
};

export default DemoMenu;
