import { rootContext } from "../context/rootProvider";
import React, { useContext, useState, useRef } from "react";
import { useHomeStyles } from "../styles/home";
import SideBar from "../components/sideBar";
import UploadData from "../components/uploadData";
import SearchVideo from "../components/searchVideo";
import { TypeDialogConfigs } from "../types";
import { imgUpload, dropTable } from "../http/index";
import { TextField } from "@material-ui/core";

const fileType = "zip";

const getObjectURL = (file: File) => {
  let url = null;
  if (window.webkitURL != undefined) {
    // webkit or chrome
    url = window.webkitURL.createObjectURL(file);
  } else if (window.URL != undefined) {
    // mozilla(firefox)
    url = window.URL.createObjectURL(file);
  }
  return url;
};

const Home = () => {
  const classes = useHomeStyles();
  const {
    setDialog,
    setGlobalLoading,
    closeDialog,
    openSnackbar,
    closeSnackbar,
  } = useContext(rootContext);
  const [activeItem, setActiveItem] = useState<"upload" | "search">("upload"); // upload or search
  const [isDataReady, setIsDataReady] = useState(false);
  const videoUploadRef = useRef<HTMLInputElement>(null!);
  const [filePath, setFilePath] = useState("");

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFilePath(e.target.value);
  };

  const dialogConfigs: TypeDialogConfigs = {
    open: true,
    type: "notice",
    params: {
      title: "Existing data set will be removed",
      component: (
        <>
          <p style={{ marginBottom: "16px" }}>
            Once you upload a new data set, the previous one will be removed.
          </p>
          <TextField
            label="Input file path"
            fullWidth={true}
            inputRef={videoUploadRef}
          ></TextField>
        </>
      ),
      confirm: () => handleFacesUpload(),
      cancel: closeDialog,
    },
  };

  const handleFacesUpload = async () => {
    closeDialog();
    setGlobalLoading(true);
    try {
      // await dropTable();
      const res = await imgUpload(videoUploadRef.current.value);
      console.log(res);
      // openSnackbar();
    } catch (error) {
      console.log(error);
    } finally {
      setGlobalLoading(false);
    }
  };

  return (
    <div className={classes.root}>
      <SideBar
        activeItem={activeItem}
        setActiveItem={setActiveItem}
        isDataReady={isDataReady}
      />
      <div className={classes.content}>
        {activeItem === "upload" ? (
          <UploadData
            setDialog={setDialog}
            dialogConfigs={dialogConfigs}
            checkDataReady={isDataReady}
            fileType={fileType}
          />
        ) : (
          <SearchVideo />
        )}
      </div>
    </div>
  );
};

export default Home;
