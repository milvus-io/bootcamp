import React, { useRef, useContext } from 'react';
import CloudUploadIcon from '@material-ui/icons/CloudUpload';
import RestoreFromTrashIcon from '@material-ui/icons/RestoreFromTrash';
import { makeStyles } from "@material-ui/core/styles";
import { queryContext } from '../contexts/QueryContext';

const useStyles = makeStyles((theme) => ({
  root: {
    display: 'flex',
    flexDirection: 'column',
    flex: " 0 0 80px",
    background: "#000",
    alignItems: "center",
    padding: "20px 0",

    [theme.breakpoints.down('xs')]: {
      flexDirection: 'row',
      alignItems: 'center',
      padding: '0 20px'
    }
  },
  uploaderBtn: {
    color: '#fff',
    position: 'relative',
    width: "80px",
    height: '80px',
    marginTop: '20px',
    fontSize: '16px',
    cursor: 'pointer',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',

    [theme.breakpoints.down('xs')]: {
      marginTop: 0,
      marginLeft: '20px'
    },

    '&:hover': {
      color: '#06AFF2'
    }
  },
  uploader: {
    opacity: 0,
    position: 'absolute',
    width: '100%',
    height: '100%',
    left: 0,
    top: 0,
    cursor: 'pointer',
    zIndex: 10
  }
}));


const Menu = () => {
  const classes = useStyles();
  const { load, openSnackBar, dropTable } = useContext(queryContext);

  const hanldeFileChange = async () => {
    const file = inputRef.current.files[0];
    const fileType = file.name.split('.')[1];
    if (!file || fileType !== 'csv') {
      openSnackBar("type error");
      return;
    }

    const fd = new FormData();
    fd.append('file', file);

    try {
      const { data } = await load(fd);
      if (data[0].status) {
        openSnackBar(data[0].msg);
      }
    } catch (error) {
      console.log(error);
    } finally {
      inputRef.current.value = '';
    }
  };

  const hanldeDropTable = async () => {
    try {
      const { data } = await dropTable();
      const { code, msg } = data[0];
      if (code) {
        openSnackBar(msg);
      } else {
        openSnackBar(msg);
      }

    } catch (error) {
      console.log(error);
    }
  };

  const inputRef = useRef(null);
  return (
    <div className={classes.root}>
      <img src="/favicon.png" alt="logo" style={{ width: "40px", height: '40px' }}></img>

      <div className={classes.uploaderBtn}>
        <CloudUploadIcon />
        <p>Upload</p>
        <input type="file" ref={inputRef} className={classes.uploader} onChange={hanldeFileChange} />
      </div>

      <div className={classes.uploaderBtn} onClick={hanldeDropTable}>
        <RestoreFromTrashIcon />
        <p>Delete</p>
      </div>



    </div>
  );
};

export default Menu;