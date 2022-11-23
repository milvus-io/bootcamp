import { Button, makeStyles } from "@material-ui/core";
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
  fileType: string;
};

const UploadData = (props: Props): JSX.Element => {
  const { setDialog, dialogConfigs, fileType } = props;
  const classes = useStyle();

  return (
    <div className={classes.uploadWrapper}>
      <div className={classes.content}>
        <div className={classes.icon}>
          <BackupIcon className={classes.uploadIcon} />
        </div>
        <h2>Upload Data Set</h2>
        <p>
          Upload a folder of <strong>{fileType}</strong> files. The maximum file
          size is 20MB.
        </p>
        <p>This process will take about 1 minute.</p>
        <div className={classes.btnWrapper}>
          <Button
            variant="contained"
            onClick={() => setDialog(dialogConfigs)}
            classes={{ root: classes.uploadBtn }}
          >
            Upload Data Set
          </Button>
        </div>
      </div>
    </div>
  );
};

export default UploadData;
