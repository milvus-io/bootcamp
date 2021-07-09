import React, { useState, createContext } from "react";
import { makeStyles, Theme, CircularProgress } from "@material-ui/core";
import Snackbar, { SnackbarOrigin } from "@material-ui/core/Snackbar";
import { uploadSet, dropSet, search, count, TABLE_NAME } from "../utils/http";

export const rootContext = createContext<any>({});
const useStyles = makeStyles((theme: Theme) => ({
  loading: {
    position: "fixed",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: "rgba(0, 0, 0, 0.5)",
  },
}));

const { Provider } = rootContext;

const ContextProvider = (props: { children: any }) => {
  const classes = useStyles();

  const [snackbarConfig, setSnackbarCong] = useState({
    open: false,
    message: "",
  });

  const [loading, setLoading] = useState(false);

  // const [dialogConfig, setDialogCong] = useState({
  //   open: false,
  //   content: <></>,
  // });

  const handleOpenSnackbar = (message: string) => {
    setSnackbarCong({
      open: true,
      message,
    });
  };

  const handleCloseSnackbar = () => {
    setSnackbarCong({
      open: false,
      message: "",
    });
  };

  return (
    <>
      <Provider
        value={{
          openSnackbar: handleOpenSnackbar,
          closeSnackbar: handleCloseSnackbar,
          uploadSet,
          dropSet,
          search,
          count,
          tableName: TABLE_NAME,
          loading,
          setLoading,
        }}
      >
        <Snackbar
          anchorOrigin={{ vertical: "top", horizontal: "left" }}
          open={snackbarConfig.open}
          onClose={handleCloseSnackbar}
          message={snackbarConfig.message}
        />
        {props.children}
      </Provider>
      {loading && (
        <div className={classes.loading}>
          <CircularProgress />
        </div>
      )}
    </>
  );
};

export default ContextProvider;
