import { rootContext } from "../context/rootProvider";
import { useContext, useState, useRef } from "react";
import { useHomeStyles } from "../styles/home";
import SideBar from "../components/sideBar";
import UploadData from "../components/uploadData";
import SearchVideo from "../components/searchVideo";
import { TypeDialogConfigs } from "../types";
import { imgUpload, dropTable } from "../http/index";
import { TextField } from "@material-ui/core";

const fileType = "jpg";

const Home = () => {
  const classes = useHomeStyles();
  const { setDialog, setGlobalLoading, closeDialog, openSnackbar } =
    useContext(rootContext);
  const [activeItem, setActiveItem] = useState<"upload" | "search">("upload"); // upload or search
  const [isDataReady, setIsDataReady] = useState(false);
  const videoUploadRef = useRef<HTMLInputElement>(null!);

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
    const filePath = videoUploadRef.current.value;
    if (!filePath) {
      openSnackbar("Please input file path", "error");
      return;
    }
    closeDialog();
    setGlobalLoading(true);
    try {
      await dropTable();
      const {
        data: { msg, status },
      } = await imgUpload(filePath);
      if (status) {
        openSnackbar(msg, "success");
        setIsDataReady(true);
        setActiveItem("search");
      } else {
        openSnackbar("Data loading failed", "error");
      }
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
