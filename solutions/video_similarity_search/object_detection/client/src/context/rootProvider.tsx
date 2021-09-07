import Rreact, { createContext, useState } from "react";
import { Snackbar, Dialog } from "@material-ui/core";
import MuiAlert, { AlertProps } from "@material-ui/lab/Alert";
import { TypeSnackbarConfig, TypeSnackbarType } from "../types";

const rootContext = {};

function Alert(props: AlertProps) {
  return <MuiAlert elevation={6} variant="filled" {...props} />;
}

const RootProvider = (props: any) => {
  const { Provider } = createContext(rootContext);
  const [dialogController, setDialogController] = useState(false);
  const [snackbarConfig, setSnackbarConfig] = useState<TypeSnackbarConfig>({
    open: false,
    anchorOrigin: { vertical: "top", horizontal: "right" },
    message: "",
    type: "info",
  });

  const openDialog = () => {
    setDialogController(true);
  };
  const closeDialog = () => {
    setDialogController(false);
  };

  const openSnackbar = (message: string, type: TypeSnackbarType) => {
    setSnackbarConfig({
      open: true,
      anchorOrigin: { vertical: "top", horizontal: "right" },
      message,
      type,
    });
  };
  const closeSnackbar = () => {
    setSnackbarConfig({
      open: false,
      anchorOrigin: { vertical: "top", horizontal: "right" },
      message: "",
      type: "info",
    });
  };

  return (
    <Provider
      value={{
        openDialog,
        closeDialog,
        openSnackbar,
        closeSnackbar,
      }}
    >
      <Dialog open={dialogController}></Dialog>
      <Snackbar
        anchorOrigin={snackbarConfig.anchorOrigin}
        open={snackbarConfig.open}
      >
        <Alert onClose={closeSnackbar} severity={snackbarConfig.type}>
          {snackbarConfig.message}
        </Alert>
      </Snackbar>
      {props.children}
    </Provider>
  );
};

export default RootProvider;
