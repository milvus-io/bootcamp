import { createContext, useState } from "react";
import {
  Snackbar,
  Dialog,
  DialogActions,
  Button,
  DialogTitle,
  DialogContent,
} from "@material-ui/core";
import MuiAlert, { AlertProps } from "@material-ui/lab/Alert";
import { TypeSnackbarConfig, TypeSnackbarType, TypeDialogType } from "../types";
import GlobalLoading from "../components/loading";

const rootContent: {
  setDialog: React.Dispatch<React.SetStateAction<any>>;
  openSnackbar: (message: string, type: TypeSnackbarType) => void;
  closeSnackbar: () => void;
  setGlobalLoading: React.Dispatch<React.SetStateAction<boolean>>;
  closeDialog: () => void;
} = {
  setDialog: () => {},
  openSnackbar: () => {},
  closeSnackbar: () => {},
  setGlobalLoading: () => {},
  closeDialog: () => {},
};

function Alert(props: AlertProps) {
  return <MuiAlert elevation={6} variant="filled" {...props} />;
}

const DefaultDialogConfigs: TypeDialogType = {
  open: false,
  bgcolor: "#fff",
  type: "notice",
  params: {
    title: "",
    component: <></>,
    confirm: () => {},
    cancel: () => {},
  },
};

export const rootContext = createContext(rootContent);

const RootProvider = (props: any) => {
  const { Provider } = rootContext;
  const [dialog, setDialog] = useState(DefaultDialogConfigs);
  const [snackbar, setSnackbar] = useState<TypeSnackbarConfig>({
    open: false,
    anchorOrigin: { vertical: "top", horizontal: "right" },
    message: "",
    type: "info",
  });

  const [globalLoading, setGlobalLoading] = useState(false);

  const openSnackbar = (message: string, type: TypeSnackbarType) => {
    setSnackbar({
      open: true,
      anchorOrigin: { vertical: "top", horizontal: "right" },
      message,
      type,
    });
  };
  const closeSnackbar = () => {
    const value = { ...snackbar, open: false };
    setSnackbar(value);
  };

  const closeDialog = () => {
    setDialog(Object.assign(DefaultDialogConfigs, { open: false }));
  };

  return (
    <Provider
      value={{
        setDialog,
        openSnackbar,
        closeSnackbar,
        setGlobalLoading,
        closeDialog,
      }}
    >
      <Dialog open={dialog.open} style={{ background: dialog.bgcolor }}>
        <DialogTitle id="form-dialog-title">{dialog.params.title}</DialogTitle>
        <DialogContent>{dialog.params.component}</DialogContent>

        <DialogActions>
          <Button onClick={dialog.params.cancel} color="primary">
            Cancel
          </Button>
          <Button onClick={dialog.params.confirm} color="primary">
            Confirm
          </Button>
        </DialogActions>
      </Dialog>
      <Snackbar
        anchorOrigin={snackbar.anchorOrigin}
        open={snackbar.open}
        autoHideDuration={3000}
        onClose={closeSnackbar}
      >
        <Alert onClose={closeSnackbar} severity={snackbar.type}>
          {snackbar.message}
        </Alert>
      </Snackbar>
      {props.children}
      <GlobalLoading open={globalLoading} />
    </Provider>
  );
};

export default RootProvider;
