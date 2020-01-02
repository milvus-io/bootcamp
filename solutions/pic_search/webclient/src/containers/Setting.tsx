import React, { useState, useEffect, useContext, useRef } from "react";
import { queryContext } from "../contexts/QueryContext";
import { createStyles, Theme, makeStyles } from "@material-ui/core/styles";
import TextField from "@material-ui/core/TextField";
import Fab from "@material-ui/core/Fab";
import AddIcon from "@material-ui/icons/Add";
import CloseIcon from "@material-ui/icons/Close";
import Slider from "@material-ui/core/Slider";
import { DropzoneArea } from "material-ui-dropzone";
import SeperatLine from "../components/SeperatLine";
import { baseColor } from "../utils/color";
import Logo from "./Logo.svg";
import { delayRunFunc } from "../utils/Helper";

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    setting: {
      display: "flex",
      flexDirection: "column",
      minWidth: "400px",
      padding: "60px 20px",
      borderWidth: "1px",
      backgroundColor: "#1F2023",
      color: "#E4E4E6",
      overflowY: "auto"
    },
    header: {
      marginBottom: "30px",
      display: "flex",
      flexDirection: "column",
      alignItems: "center"
    },
    configHead: {
      marginBottom: "30px",
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center"
    },
    config: {
      fontSize: "24px",
      color: "#FAFAFA"
    },
    clear: {
      color: baseColor,
      fontSize: "18px",
      cursor: "pointer"
    },
    imageSet: {},
    counts: {
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      marginBottom: "30px",
      color: "#FAFAFA"
    },
    currTotal: {
      fontSize: "12px"
    },
    setPath: {
      display: "flex",
      justifyContent: "start",
      alignItems: "center",
      marginBottom: "30px"
    },
    customInput: {
      margin: "0 20px 0 0 !important",
      color: "blue !important"
    },
    customFab: {
      color: "#fff",
      backgroundColor: baseColor,
      width: "36px",
      height: "36px",
      "&:hover": {
        backgroundColor: baseColor
      }
    },
    customDeleteFab: {
      position: "absolute",
      top: "5px",
      right: "5px",
      color: "#fff",
      backgroundColor: "#666769",
      width: "24px",
      height: "24px",
      minHeight: "0px",
      "&:hover": {
        backgroundColor: "#666769"
      }
    },
    customDelete: {
      color: "#A7A7AF",
      width: "18px",
      height: "18px"
    },
    customIcon: {
      color: "#fff",
      backgroundColor: baseColor,
      width: "20px",
      height: "20px"
    },
    customSlider: {
      color: baseColor,
      marginBottom: "30px"
    },
    thumb: {
      width: "16px",
      height: "16px"
    },
    track: {
      height: "4px",
      borderRadius: "10px"
    },
    upload: {
      display: "flex",
      justifyContent: "center",
      alignItems: "center"
    },
    benchImage: {
      width: "400px",
      height: "250px",
      position: "relative"
    },
    dropzoneContainer: {
      backgroundColor: "transparent",
      width: "250px",
      height: "250px",
      borderRadius: "10px",
      border: "solid .5px #C8C8C8",
      display: "flex",
      justifyContent: "center",
      alignItems: "center"
    },
    dropzoneText: {
      fontSize: "14px",
      color: "#B3B4B5",
      marginBottom: "30px"
    },
    notchedOutline: {
      borderWidth: ".5px",
      borderColor: "#838385 !important"
    },
    formLabel: {
      color: "#fff"
    },
    controlLabel: {
      color: "#838385"
    }
  })
);

const Setting = (props: any) => {
  const { process, train, count, search, clearAll } = useContext(queryContext);
  const { setImages, loading, setLoading } = props;
  const classes = useStyles({});
  const [inputs, setInputs]: any = useState("");
  const [topK, setTopK]: any = useState(5);
  const [totalNum, setTotalNum]: any = useState(0);
  const [[current, total], setProcessedNum]: any = useState([0, 0]);
  const [image, setImage]: any = useState();

  const benchImage = useRef<any>(null);
  const setText = loading
    ? "Loading..."
    : totalNum
    ? `${totalNum} images in this set`
    : "No image in this set";

  const reader = new FileReader();
  reader.addEventListener(
    "load",
    function() {
      if (benchImage.current) {
        benchImage.current.src = reader.result;
      }
    },
    false
  );
  const _search = ({ topK, image }: any) => {
    const fd = new FormData();
    fd.set("Num", topK);
    fd.append("file", image);
    search(fd).then((res: any) => {
      const { status, data } = res || {};
      if (status === 200) {
        setImages(data);
      }
    });
  };

  const uploadImg = (file: any) => {
    setImage(file);
    reader.readAsDataURL(file);
    _search({ topK, image: file });
  };

  const onInputChange = (e: any) => {
    const val = e.target.value;
    setInputs(val);
  };

  const onTopKChange = (e: any, val: any) => {
    setTopK(val);
    if (val && image) {
      delayRunFunc({ topK: val, image }, _search, 300);
    }
  };
  const _keepProcess = () => {
    process().then((res: any) => {
      const { data, status } = res;
      if (status === 200) {
        const [_current, _total] = data
          .split(",")
          .map((item: any) => Number.parseInt(item.split(":")[1]));
        setProcessedNum([_current, _total]);
        if (_current !== _total) {
          setTimeout(() => _keepProcess(), 1000);
        } else {
          setTimeout(() => {
            count().then((res: any) => {
              const { data, status } = res;
              if (status === 200) {
                setTotalNum(data);
                setLoading(false);
              }
            });
          }, 3000);
        }
      }
    });
  };
  const uploadImgPath = () => {
    train({ File: inputs }).then((res: any) => {
      if (res.status === 200) {
        setLoading(true);
        setTimeout(() => {
          setInputs("");
          _keepProcess();
        }, 1000);
      }
    });
  };

  const clear = () => {
    clearAll().then((res: any) => {
      if (res.status === 200) {
        setProcessedNum([0, 0]);
        setTotalNum(0);
        setImage();
        setImages([]);
      }
    });
  };

  useEffect(() => {
    count().then((res: any) => {
      const { data, status } = res || {};
      if (status === 200) {
        setTotalNum(data);
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className={classes.setting}>
      <div className={classes.header}>
        <img src={Logo} width="150px" alt="logo" />
        <p>Image Search Demo</p>
      </div>
      <div className={classes.configHead}>
        <h4 className={classes.config}>Config</h4>
        <h4 className={classes.clear} onClick={clear}>
          CLEAR ALL
        </h4>
      </div>
      <SeperatLine title={`IMAGE SET`} style={{ marginBottom: "20px" }} />
      <div className={classes.imageSet}>
        <div className={classes.counts}>
          <p style={{ color: loading ? baseColor : "#fff" }}>{setText}</p>
          <h3 className={classes.currTotal}>{`${current}/${total}`}</h3>
        </div>
        <div className={classes.setPath}>
          <TextField
            classes={{ root: classes.customInput }}
            label=""
            variant="outlined"
            value={inputs}
            onChange={onInputChange}
            InputLabelProps={{
              shrink: true,
              classes: {
                root: classes.controlLabel,
                focused: classes.controlLabel
              }
            }}
            margin="normal"
            InputProps={{
              style: {
                textAlign: "left",
                width: "340px",
                height: "40px"
              },
              classes: {
                notchedOutline: classes.notchedOutline,
                root: classes.formLabel
              },
              placeholder: "path/to/your/images"
            }}
          />
          <Fab
            classes={{
              root: classes.customFab,
              focusVisible: classes.customFab
            }}
          >
            <AddIcon
              onClick={uploadImgPath}
              classes={{ root: classes.customIcon }}
            />
          </Fab>
        </div>
        <SeperatLine title={`TOP K(1－100)`} style={{ marginBottom: "20px" }} />
        <div className={classes.counts}>
          <p>{`show top ${topK} results`}</p>
        </div>
        <Slider
          min={1}
          max={100}
          value={topK}
          onChange={onTopKChange}
          classes={{
            root: classes.customSlider,
            track: classes.track,
            rail: classes.track,
            thumb: classes.thumb
          }}
        />
      </div>
      <SeperatLine title={`ORIGINAL IMAGE`} style={{ marginBottom: "50px" }} />
      <div className={classes.upload}>
        {image ? (
          <div className={classes.benchImage}>
            <img
              ref={benchImage}
              className={classes.benchImage}
              src={image}
              alt="..."
            />
            <Fab
              color="primary"
              aria-label="add"
              size="small"
              classes={{ root: classes.customDeleteFab }}
            >
              <CloseIcon
                onClick={() => {
                  setImage();
                  setImages([]);
                }}
                classes={{ root: classes.customDelete }}
              />
            </Fab>
          </div>
        ) : (
          <DropzoneArea
            acceptedFiles={["image/*"]}
            filesLimit={1}
            dropzoneText={`click to upload / drag a image here`}
            onDrop={uploadImg}
            dropzoneClass={classes.dropzoneContainer}
            showPreviewsInDropzone={false}
            dropzoneParagraphClass={classes.dropzoneText}
            // maxFileSize={} bit
          />
        )}
      </div>
    </div>
  );
};

export default Setting;
