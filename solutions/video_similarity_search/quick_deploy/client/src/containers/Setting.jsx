import React, { useState, useEffect, useContext, useRef } from "react";
import { queryContext } from '../contexts/QueryContext';
import AddIcon from "@material-ui/icons/Add";
import { genID } from '../utils/Helper';
import FileDrop from 'react-file-drop';
import clsx from "clsx";
import { makeStyles } from "@material-ui/core/styles";
import useMediaQuery from "@material-ui/core/useMediaQuery";
import WarnningIcon from '@material-ui/icons/Warning';
import DeleteIcon from "@material-ui/icons/Delete";
import { TextField, Fab } from "@material-ui/core";
import SeperatLine from '../components/SeperatLine';
import { baseColor } from "../utils/color";
import './Setting.css';

const Setting = (props) => {
  const isMobile = !useMediaQuery("(min-width:1000px)");
  const { results, setResults } = props;
  const { search, setNavTitle, searchParams, setSearchParams, videosLoad, showNote, videosDrop, queryStatus, getCount } = useContext(queryContext);
  const useStyles = makeStyles({
    setting: {
      position: "relative",
      minWidth: isMobile ? "auto" : "303px",
      maxWidth: isMobile ? "auto" : "20%",
      padding: isMobile ? "20px" : "20px 20px 0 20px",
      borderWidth: 0,
      borderRight: `solid 1px ${results.length ? "" : '!important'} `,
      borderImage: 'linear-gradient(#fff, rgba(250,250,250,0)) 0 100%',
      color: "#E4E4E6",
      overflowY: "auto",
      backgroundColor: "#1F2023",
    },
    addWrapper: {
      width: "calc(100%-2px)",
      marginBottom: '5px',
      height: '15vh',
      background: 'rgba(255,255,255,0.1)',
      border: '1px solid rgba(176,176,185,1)',
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      cursor: 'pointer'
    },
    imageWrapper: {
      maxWidth: "100%",
      marginBottom: '5px',
      opacity: .3,
      position: "relative",
    },
    selectedImage: {
      border: '1px solid rgba(255,255,255,1)',
      opacity: 1
    },
    delete: {
      color: '#fff',
      background: 'red',
      cursor: 'pointer'
    },
    dragEnter: {
      border: '2px solid rgba(63, 156, 209, 1) !important',
      color: 'rgba(63, 156, 209, 1) !important',
    },
    MolecularInput: {
      flexGrow: 1,
      color: "blue !important",
      margin: 0
    },
    controlLabel: {
      color: "#838385",
    },
    notchedOutline: {
      borderWidth: ".5px",
      borderColor: "#838385 !important",
    },
    formLabel: {
      color: "#fff",
    },
    inputWrapper: {
      width: '100%',
      marginBottom: '20px',
      display: 'flex',
      alignItems: 'center'
    },
    customFab: {
      marginLeft: '16px',
      color: "#fff",
      backgroundColor: baseColor,
      width: "20px",
      height: "20px",
      minHeight: '20px'
    },
    customIcon: {
      fontSize: '14px'
    },
    counts: {
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      marginBottom: "30px",
      color: "#FAFAFA",
    },
    currTotal: {
      fontSize: "12px",
    },
  });
  const classes = useStyles({});
  const [videosPath, setVideosPath] = useState("");
  const [deleteID, setDeleteID] = useState('');
  const uploader = useRef(null);
  const isSubscription = useRef(true);
  const FileUploader = useRef(null);
  const [count, setCount] = useState(0);
  const [loading, setLoading] = useState(false);

  const setText = loading
    ? "Loading..."
    : count
      ? `${count} Frames in this set`
      : "No Videos in this set";

  const changeImg = (curr) => {
    setSearchParams(searchParams => ({ ...searchParams, curr }));
  };
  const delHistory = (id) => {
    let curr = searchParams.curr;
    const index = searchParams.history.indexOf(curr);
    if (searchParams.curr.id === id) {
      curr = searchParams.history[index + 1] || searchParams.history[index - 1] || {
        file: "",
        data: "",
        id: ""
      };
    }
    const arr = searchParams.history.filter((t) => t.id !== id);
    setSearchParams({ curr, history: arr });
  };
  const _search = async (imgSrc) => {
    setLoading(true);
    setNavTitle('SEARCHING...');
    try {
      const { status, data } = await search(imgSrc);
      if (status === 200) {
        setNavTitle(`${data.length} RESULTS`);
        setResults(data);
      }
    } catch (error) {
      console.log(error);
    } finally {
      setLoading(false);
    }

  };
  const clickUpload = () => {
    if (FileUploader.current) {
      FileUploader.current.onchange = _clickAddSearchImg;
      FileUploader.current.click();
    }
  };
  const _readFile = (file) => {
    const reader = new FileReader();
    reader.addEventListener("load", function () {
      const { history } = searchParams;
      const newOne = { file, data: reader.result, id: genID() };
      history.splice(0, 0, newOne);
      setSearchParams({ history, curr: newOne });
    }, false);
    if (file) {
      reader.readAsDataURL(file);
    }
  };
  const _addSearchImg = (e) => {
    setNavTitle('UPLOADING...');
    const file = e.dataTransfer.files[0];
    _readFile(file);
  };
  const _clickAddSearchImg = (e) => {
    const file = e.target.files[0];
    _readFile(file);
  };

  const handleInputChange = e => {
    const value = e.target.value;
    setVideosPath(value);
  };

  // upload Video collection
  const setUploadPath = async () => {
    if (!videosPath) {
      showNote('Please input path!');
      return;
    }
    setLoading(true);
    try {
      const { status } = await videosLoad({ File: videosPath });
      if (status === 200) {
        await getFrameCount();
      }
    } catch (error) {
      console.log(error);
    } finally {
      setLoading(false);
    }
  };

  // clear Video collection
  const clear = async () => {
    try {
      const { status, data: { message } } = await videosDrop();
      if (status === 200) {
        setCount(0);
        showNote(message);
      }
    } catch (error) {
      console.log(error);
    }
  };

  const progress = async () => {
    try {
      const { status, data: { current, total } } = await queryStatus();
      if (status === 200) {
        return {
          current, total
        };
      }
    } catch (error) {
      console.log(error);
    }
  };

  const getFrameCount = async () => {
    try {
      const { status, data } = await getCount();
      if (status === 200) {
        setCount(data);
      }
    } catch (error) {
      console.log(error);
    }

  };

  useEffect(() => {
    getFrameCount();
  }, []);

  useEffect(() => {
    isSubscription.current = true;
    if (searchParams.curr.file) {
      _search(searchParams.curr.file);
    }

    const Uploader = uploader.current || document.createElement('div');
    const onDragEnter = () => { Uploader.classList.add('drag-enter'); };
    const onDragLeave = () => { Uploader.classList.remove('drag-enter'); };
    document.body.addEventListener('dragenter', onDragEnter);
    document.body.addEventListener('dragleave', onDragLeave);
    document.body.addEventListener('drop', _addSearchImg);
    return () => {
      document.body.removeEventListener('dragenter', onDragEnter);
      document.body.removeEventListener('dragleave', onDragLeave);
      document.body.removeEventListener('drop', _addSearchImg);
      isSubscription.current = false;
    };
    // eslint-disable-next-line
  }, [searchParams.curr]);
  return (
    <div className={classes.setting}>
      <FileDrop>
        <div className={classes.addWrapper} ref={uploader} onClick={clickUpload} onMouseEnter={() => uploader.current.classList.add('drag-enter')} onMouseLeave={() => uploader.current.classList.remove('drag-enter')}>
          <AddIcon />
          <input type="file" style={{ display: 'none' }} ref={FileUploader} multiple />
        </div>
      </FileDrop>
      {
        searchParams.history.map((item, index) => {
          const isSelected = item.id === searchParams.curr.id;
          const isDelete = item.id === deleteID;
          return (
            <div key={index} className={clsx(classes.imageWrapper, isSelected ? classes.selectedImage : "")} onClick={() => changeImg(item)} onMouseEnter={() => setDeleteID(item.id)} onMouseLeave={() => { item.id === deleteID && setDeleteID(""); }}>
              <img style={{ width: '100%' }} src={item.data} alt="" />
              {isDelete && <div style={{ position: 'absolute', top: 0, right: 0 }}><DeleteIcon classes={{ root: classes.delete }} onClick={(e) => { e.stopPropagation(); delHistory(item.id); return false; }} /></div>}
            </div>
          );
        })
      }

      <SeperatLine
        title={`CONFIG`}
        end="CLEAR ALL"
        onEndClick={clear}
        style={{ margin: "30px 0" }}
      />
      <div className={classes.counts}>
        <p style={{ color: loading ? baseColor : "#fff" }}>{setText}</p>
      </div>
      <div className={classes.inputWrapper}>

        <TextField
          classes={{ root: classes.MolecularInput }}
          label=""
          variant="outlined"
          value={videosPath || ""}
          onChange={handleInputChange}
          InputLabelProps={{
            shrink: true,
            classes: {
              root: classes.controlLabel,
              focused: classes.controlLabel,
            },
          }}
          margin="normal"
          InputProps={{
            style: {
              textAlign: "left",
              width: "100%",
              height: "40px",
            },
            classes: {
              notchedOutline: classes.notchedOutline,
              root: classes.formLabel,
            },
            placeholder: "please input your video path",
            onKeyPress: (e) => {
              if (e.key === "Enter") {
                setUploadPath();
              }
            },
          }}
        />
        <Fab
          classes={{
            root: classes.customFab,
            focusVisible: classes.customFab,
          }}
        >
          <AddIcon
            onClick={setUploadPath}
            classes={{ root: classes.customIcon }}
          />
        </Fab>
      </div>

    </div >
  );
};

export default Setting;
