import React, { useState, useEffect, useContext, useRef } from "react";
import Masonry from "react-masonry-component";
import WarnningIcon from "@material-ui/icons/Warning";
import { queryContext } from "../contexts/QueryContext";
import { makeStyles } from "@material-ui/core/styles";
import useMediaQuery from "@material-ui/core/useMediaQuery";
import AddIcon from "@material-ui/icons/Add";
// import DeleteIcon from "@material-ui/icons/Delete";
var GifPlayer = require("react-gif-player");

const _calPercent = ({ percent, state }) => {
  return (state !== "predict"
    ? (percent * 100) / 2
    : 50 + (percent * 100) / 2
  ).toFixed(2);
};

const regex = new RegExp(/\.gif$/i);
const masonryOptions = {
  transitionDuration: 500,
  initLayout: true,
  resize: true
};

const group_size = 2;
const Libarary = () => {
  const {
    queryLibrary,
    navTitle,
    setNavTitle,
    upload,
    queryStatus
    // delVideo
  } = useContext(queryContext);
  const isMobile = !useMediaQuery("(min-width:1000px)");
  const [results, setResults] = useState([]);
  // const [selectedID, setSelectedID] = useState("");
  const [loadingPercent, setLoadingPercent] = useState(0);
  const [uploadQueue, setUploadQueue] = useState([]);
  const [page, setPage] = useState(0);

  const useStyles = makeStyles({
    root: {
      flexGrow: 1,
      overflowX: "hidden",
      overflowY: "auto",
      padding: isMobile ? "10px" : "20px",
      display: "flex",
      flexDirection: "column"
    },
    container: {
      width: "100%",
      position: "relative"
    },
    imgWrapper: {
      width: "16.5%",
      border: "solid 1px transparent",
      minHeight: "40px",
      display: "block",
      position: "relative"
    },
    cover: {
      position: "absolute",
      top: 0,
      right: 0,
      width: `${100 - loadingPercent}%`,
      height: "calc(100% - 2px)",
      backgroundColor: "rgba(79,196,249,0.5)"
    },
    percent: {
      position: "absolute",
      bottom: "5px",
      right: "5px",
      color: "#fff",
      textShadow: `black 0.1em 0.1em 0.2em`
    },
    selected: {
      border: "solid 1px red"
    },
    delete: {
      color: "#fff",
      background: "red",
      cursor: "pointer"
    },
    addWrapper: {
      width: "16.5%",
      height: "15vh",
      background: "rgba(255,255,255,0.1)",
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      border: "1px solid rgba(176,176,185,1)",
      color: "#fff",
      cursor: "pointer"
    }
  });
  const classes = useStyles({});
  const uploader = useRef(null);
  const FileUploader = useRef(null);
  const uploaderID = useRef(null);
  const GifUploading = useRef("");
  const TotalContainer = useRef(0);
  const isSubscription = useRef(true);
  const should_query = useRef(true);
  const Root = useRef(null);
  const cdd_results = useRef([]);
  const render_num = useRef(0);

  // const onMouseOver = id => setSelectedID(id);
  // const onMouseLeave = id => selectedID === id && setSelectedID("");
  // const deleteGif = name => {
  //   setResults(results => results.filter(result => result.name !== name));
  //   delVideo(name).then(res => {
  //     if (res && res.status === 200 && isSubscription.current) {
  //       TotalContainer.current = TotalContainer.current - 1;
  //       setNavTitle(`${TotalContainer.current} VIDEOS IN LIBRARY`);
  //     }
  //   });
  // };
  const clickUpload = () => {
    if (FileUploader.current) {
      FileUploader.current.onchange = e => {
        const files = [...e.target.files].filter(item => regex.test(item.name));
        setUploadQueue(files);
      };
      FileUploader.current.click();
    }
  };

  const _loadResults = () => {
    if (
      cdd_results.current.length > 0 &&
      render_num.current % group_size === 0
    ) {
      const cdds = cdd_results.current.splice(0, group_size);
      setResults(results => [...results, ...cdds]);
    }
  };

  const isFirstRun = useRef(true);
  useEffect(() => {
    isSubscription.current = true;
    const query = async () => {
      should_query.current = false;
      setNavTitle(`Fetching New Data...`);
      queryLibrary({ page }).then(res => {
        if (res && res.status === 200 && isSubscription.current) {
          const { Data, Total } = res.data;
          if (Total === 0) {
            should_query.current = false;
          }
          if (isFirstRun.current) {
            isFirstRun.current = false;
            TotalContainer.current = Total;
          }
          cdd_results.current = Data;
          render_num.current = 0;
          _loadResults();
          setNavTitle(`${TotalContainer.current} VIDEOS IN LIBRARY`);
          should_query.current = true;
        }
      });
    };
    if (should_query.current && cdd_results.current.length === 0) {
      query();
    }
    return () => {
      isSubscription.current = false;
    };
    //eslint-disable-next-line
  }, [page]);
  // bind drag and drop event
  useEffect(() => {
    isSubscription.current = true;
    const _upload = async e => {
      const files = [...e.dataTransfer.files].filter(item =>
        regex.test(item.name)
      );
      if (files && files.length > 0 && isSubscription.current) {
        setUploadQueue(files);
      }
    };
    const Uploader = uploader.current || document.createElement("div");
    const _onMouseEnter = e => {
      if (uploader.current) {
        uploader.current.classList.add("drag-enter");
      }
    };
    const _onMouseLeave = e => {
      if (uploader.current) {
        uploader.current.classList.remove("drag-enter");
      }
    };
    document.body.addEventListener("drop", _upload);
    document.body.addEventListener("dragenter", _onMouseEnter);
    document.body.addEventListener("dragleave", _onMouseLeave);
    Uploader.addEventListener("mouseenter", _onMouseEnter);
    Uploader.addEventListener("mouseleave", _onMouseLeave);

    const _fetch = e => {
      const clientHeight = document.getElementById("content").clientHeight;
      const scrollHeight = document.getElementById("content").scrollHeight;
      const scrollTop = document.getElementById("content").scrollTop;
      if (scrollHeight === clientHeight + scrollTop && should_query.current) {
        setPage(page => page + 1);
      }
    };
    window.addEventListener("scroll", _fetch);
    window.addEventListener("mousewheel", _fetch);
    return () => {
      document.body.removeEventListener("drop", _upload);
      document.body.removeEventListener("dragenter", _onMouseEnter);
      document.body.removeEventListener("dragleave", _onMouseLeave);
      Uploader.addEventListener("mouseenter", _onMouseEnter);
      Uploader.addEventListener("mouseleave", _onMouseLeave);
      window.removeEventListener("scroll", _fetch);
      window.removeEventListener("mousewheel", _fetch);
      isSubscription.current = false;
    };
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [results]);
  useEffect(() => {
    isSubscription.current = true;
    let timeout;
    const _finishOneUpload = () => {
      TotalContainer.current = TotalContainer.current + 1;
      setNavTitle(`${TotalContainer.current} VIDEOS IN LIBRARY`);
      setResults(results => [
        { name: uploaderID.current, data: GifUploading.current },
        ...results
      ]);
      setUploadQueue(uploadQueue => uploadQueue.splice(1));
    };
    const _keepProcess = async id => {
      queryStatus(id).then(res => {
        if (isSubscription.current) {
          if (res && res.status === 200) {
            const percent = _calPercent(res.data);
            setLoadingPercent(Math.floor(percent * 100) / 100);
            percent >= 100
              ? _finishOneUpload()
              : (function() {
                  setLoadingPercent(percent);
                  timeout = setTimeout(() => {
                    _keepProcess(id);
                  }, 500);
                })();
          } else {
            setNavTitle(
              <div style={{ alignItems: "center", display: "flex" }}>
                <WarnningIcon
                  style={{ color: "yellow", marginRight: "20px" }}
                />
                <span>UPLOAD FAIL</span>
              </div>
            );
          }
        }
      });
    };
    const _uploadOneGif = async file => {
      const reader = new FileReader();
      reader.addEventListener(
        "load",
        function() {
          GifUploading.current = reader.result;
          setNavTitle("UPLOADING...");
          upload(file).then(res => {
            if (isSubscription.current) {
              if (res && res.status === 200) {
                const id = res.data.id;
                uploaderID.current = id;
                _keepProcess(id);
              } else {
                setNavTitle(
                  <div style={{ alignItems: "center", display: "flex" }}>
                    <WarnningIcon
                      style={{ color: "yellow", marginRight: "20px" }}
                    />
                    <span>UPLOAD FAIL</span>
                  </div>
                );
                setUploadQueue([]);
              }
            }
          });
        },
        false
      );
      reader.readAsDataURL(file);
    };
    if (uploadQueue.length) {
      _uploadOneGif(uploadQueue[0]);
    }
    return () => {
      timeout && clearTimeout(timeout);
      isSubscription.current = false;
    };
    //eslint-disable-next-line
  }, [uploadQueue]);

  return (
    <div className={classes.root} ref={Root}>
      <div className={classes.container}>
        <Masonry
          enableResizableChildren={true}
          className={""} // default ''
          elementType={"div"} // default 'div'
          options={masonryOptions} // default {}
          updateOnEachImageLoad={true} // default false and works only if disableImagesLoaded is false
          imagesLoadedOptions={{}} // default {}
        >
          {navTitle === "UPLOADING..." ? (
            <div className={classes.imgWrapper}>
              <GifPlayer gif={GifUploading.current} autoplay />
              {loadingPercent < 100 && (
                <>
                  <div className={classes.cover} />
                  <div className={classes.percent}>{`${loadingPercent}%`}</div>
                </>
              )}
            </div>
          ) : (
            <div
              className={classes.addWrapper}
              ref={uploader}
              onClick={() => clickUpload()}
            >
              <AddIcon />
              <input
                type="file"
                style={{ display: "none" }}
                ref={FileUploader}
                multiple
              />
            </div>
          )}
          {results.map(data => {
            // const isSelected = data.name === selectedID;
            return (
              <div
                className={`${classes.imgWrapper} ${
                  ""
                  // isSelected ? classes.selected : ""
                }`}
                key={data.name}
                // onMouseOver={() => {
                //   onMouseOver(data.name);
                // }}
                // onMouseLeave={() => {
                //   onMouseLeave(data.name);
                // }}
              >
                <GifPlayer
                  gif={data.data}
                  autoplay
                  onLoad={() => {
                    render_num.current += 1;
                    _loadResults();
                  }}
                />
                {/* {isSelected && (
                  <div style={{ position: "absolute", top: 0, right: 0 }}>
                    <DeleteIcon
                      classes={{ root: classes.delete }}
                      onClick={() => deleteGif(data.name)}
                    />
                  </div>
                )} */}
              </div>
            );
          })}
        </Masonry>
      </div>
    </div>
  );
};

export default Libarary;
