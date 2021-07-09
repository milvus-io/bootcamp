import React, { useState, useContext } from "react";
import { makeStyles, Theme, OutlinedInput } from "@material-ui/core";
import ControlPointIcon from "@material-ui/icons/ControlPoint";
import { rootContext } from "../context";

const useStyles = makeStyles((theme: Theme) => ({
  root: {
    width: "100%",
    height: "100%",
    position: "relative",
  },
  container: {
    width: "60%",
    height: "60%",
    background: "#fff",
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%,-50%)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  verticalCenter: {
    // display: "flex",
    // flexDirection: "column",
    // alignItems: "center",
  },
  titleContainer: {
    marginBottom: "40px",
    textAlign: "center",
  },
  iconWrapper: {
    width: "100px",
    height: "100px",
    borderRadius: "50%",
    border: "1px solid #000",
    display: "inline-block",
    textAlign: "center",
    marginBottom: "40px",

    "& svg": {
      width: "50px",
      height: "50px",
      marginTop: "25px",
    },
  },
  alert: {
    fontSize: "16px",
    marginBottom: "40px",
  },
  title: {
    fontSize: "24px",
    fontWeight: "bold",
  },
  hidden: {
    visibility: "hidden",
    opacity: 0,
    width: 0,
    height: 0,
  },
  inputWrapper: {
    display: "flex",
    marginBottom: "40px",
  },
  input: {
    height: "40px",
  },
  addIconWrapper: {
    fontSize: "24px",
    display: "inline-block",
    lineHeight: "40px",
    marginLeft: "16px",
    cursor: "pointer",

    "&:hover": {
      color: "#6674c4",
    },
  },
}));

type PropsType = {
  setTab: React.Dispatch<React.SetStateAction<"upload" | "search">>;
};

const Uploader: React.FC<PropsType> = ({ setTab }) => {
  const { uploadSet, dropSet, tableName, openSnackbar } =
    useContext(rootContext);
  const classes = useStyles();
  const [path, setPath] = useState({
    value: "",
    showError: false,
  });

  const handleInput = (e: React.ChangeEvent<any>) => {
    const { value } = e.target;

    setPath({
      value,
      showError: false,
    });
  };

  const handleUploadSet = async () => {
    try {
      // await dropSet({ table_name: tableName });
      const { data } = await uploadSet({
        Table: tableName,
        File: path.value,
      });
      console.log(data);
      openSnackbar();
      setTab("search");
    } catch (error) {
      console.log(error);
    }
  };

  return (
    <div className={classes.root}>
      <h1 className={classes.hidden}></h1>
      <h2 className={classes.hidden}></h2>
      <div className={classes.container}>
        <div className={classes.verticalCenter}>
          <div className={classes.titleContainer}>
            <span className={classes.iconWrapper}>
              <svg
                aria-hidden="true"
                focusable="false"
                data-prefix="fas"
                data-icon="upload"
                className="svg-inline--fa fa-upload fa-w-16 jss296"
                role="img"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 512 512"
              >
                <path
                  fill="currentColor"
                  d="M296 384h-80c-13.3 0-24-10.7-24-24V192h-87.7c-17.8 0-26.7-21.5-14.1-34.1L242.3 5.7c7.5-7.5 19.8-7.5 27.3 0l152.2 152.2c12.6 12.6 3.7 34.1-14.1 34.1H320v168c0 13.3-10.7 24-24 24zm216-8v112c0 13.3-10.7 24-24 24H24c-13.3 0-24-10.7-24-24V376c0-13.3 10.7-24 24-24h136v8c0 30.9 25.1 56 56 56h80c30.9 0 56-25.1 56-56v-8h136c13.3 0 24 10.7 24 24zm-124 88c0-11-9-20-20-20s-20 9-20 20 9 20 20 20 20-9 20-20zm64 0c0-11-9-20-20-20s-20 9-20 20 9 20 20 20 20-9 20-20z"
                ></path>
              </svg>
            </span>
            <h3 className={classes.title}>Upload Data</h3>
          </div>

          <p className={classes.alert}>
            Once you upload a new data set, the previous one will be removed.
          </p>
          <div className={classes.inputWrapper}>
            <OutlinedInput
              id="outlined-basic"
              fullWidth={true}
              onChange={handleInput}
              classes={{ root: classes.input }}
            />
            <span className={classes.addIconWrapper} onClick={handleUploadSet}>
              <ControlPointIcon />
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Uploader;
