import { DEMOS_BASE_URL } from "../http/constants";
import { Button, CircularProgress, makeStyles } from "@material-ui/core";
import BackupIcon from "@material-ui/icons/Backup";
import { TypeDialogConfigs } from "../types";

const useStyle = makeStyles((theme) => ({
  uploadWrapper: {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",

    height: "100%",
    width: "100%",
    color: "#000",
  },
  content: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    width: "592px",
    height: "464px",
    backgroundColor: "#fff",

    [theme.breakpoints.down(theme.breakpoints.values.sm)]: {
      width: "100%",
      height: "100%",
      padding: theme.spacing(2),
    },

    "& p": {
      fontSize: "12.8px",
      marginBottom: theme.spacing(2),
    },

    "& h2": {
      fontWeight: "bold",
      marginBottom: "42px",
    },
  },
  icon: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    width: "106px",
    height: "106px",
    borderRadius: "50%",
    border: "2px solid #000",
    marginBottom: theme.spacing(2),
  },
  uploadIcon: {
    fontSize: "46px",
  },
  downloadBtn: {
    color: "#12c3f4",
    textDecoration: "none",
  },
  btnWrapper: {
    marginTop: "20px",
    "& button:first-child": {
      marginRight: theme.spacing(2),
    },
  },
  loading: {
    color: "#fff",
    marginRight: theme.spacing(1),
  },
  uploadBtn: {
    background: "#12c3f4",
    color: "#fff",
    minWidth: "64px",
    boxShadow: "none",

    "&:hover": {
      background: "#65daf8",
      boxShadow: "none",
    },
  },
}));

type Props = {
  setDialog: React.Dispatch<React.SetStateAction<any>>;
  dialogConfigs: TypeDialogConfigs;
};

const UploadData = (props: any) => {
  const {
    setDialog,
    dialogConfigs,
    fileType,
    downloadPath,
    checkDataReady = false,
  } = props;
  const classes = useStyle();

  return (
    <div className={classes.uploadWrapper}>
      <div className={classes.content}>
        <div className={classes.icon}>
          <BackupIcon className={classes.uploadIcon} />
        </div>
        <h2>Upload Data Set</h2>
        <p>
          Upload a zip file with <strong>{fileType}</strong> only. The maximum
          file size is 20MB.
        </p>
        <p>This process will take about 1 minute.</p>
        <div className={classes.btnWrapper}>
          <Button>
            <a
              href={`${DEMOS_BASE_URL}/${downloadPath}`}
              className={classes.downloadBtn}
            >
              download an example file
            </a>
          </Button>
          {!checkDataReady ? (
            <Button
              variant="contained"
              onClick={() => setDialog(dialogConfigs)}
              classes={{ root: classes.uploadBtn }}
            >
              Upload Data Set
            </Button>
          ) : (
            <Button variant="contained" disabled>
              <CircularProgress size={16} classes={{ root: classes.loading }} />
              Analyzing Data
            </Button>
          )}
        </div>
      </div>
    </div>
  );
};

export default UploadData;
