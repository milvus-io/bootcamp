import React, { useState, useEffect, useRef, useCallback } from "react";
import SettingsIcon from "@material-ui/icons/Settings";
import clsx from "clsx";
import { makeStyles } from "@material-ui/core/styles";
import useMediaQuery from "@material-ui/core/useMediaQuery";
import "./style.css";
import Masonry from "react-masonry-component";

var GifPlayer = require("react-gif-player");

const masonryOptions = {
  transitionDuration: 500
};
const group_size = 2;
const Results = props => {
  const { results } = props;
  const [renderResults, setResults] = useState([]);
  const isMobile = !useMediaQuery("(min-width:1000px)");
  const useStyles = makeStyles({
    root: {
      flexGrow: 1,
      overflowX: "hidden",
      overflowY: "auto",
      padding: isMobile ? "10px" : "20px",
      display: "flex",
      flexDirection: results.length ? "column" : "row",
      color: "#fff"
    },
    container: {
      width: "100%",
      position: "relative"
    },
    imgWrapper: {
      width: "19.5%",
      minHeight: "40px",
      display: "block",
      position: "relative",
      opacity: 0.75,
      border: "solid 1px transparent"
    },
    info: {
      position: "absolute",
      left: 0,
      top: 0,
      background: "transparent",
      zIndex: 10,
      padding: "10px",
      color: "#fff",
      textShadow: `black 0.1em 0.1em 0.2em`
    }
  });
  const render_nums_ref = useRef(0);
  const render_group_size_ref = useRef(group_size);
  const classes = useStyles({});
  const _loadNewImages = useCallback(() => {
    if (render_nums_ref.current === 0) {
      setResults(results.slice(0, render_nums_ref.current + group_size));
      return;
    }
    if (
      render_nums_ref.current < results.length &&
      render_nums_ref.current % render_group_size_ref.current === 0
    ) {
      render_group_size_ref.current += group_size;
      setResults(results.slice(0, render_nums_ref.current + group_size));
    }
  }, [results]);
  useEffect(() => {
    setResults([]);
    setTimeout(() => {
      render_nums_ref.current = 0;
      render_group_size_ref.current = group_size;
      _loadNewImages();
    }, 10);
    // eslint-disable-next-line
  }, [results]);
  return (
    <div className={classes.root}>
      {renderResults.length === 0 ? (
        <div
          style={{
            fontFamily: `Roboto-Regular,Roboto`,
            fontWeight: 400,
            color: `rgba(250,250,250,1)`
          }}
        >
          <p style={{ marginBottom: "20px" }}>
            To search, drop an image or video file.
          </p>
          {/* <div
            style={{
              display: `flex`,
              justifyContent: "center",
              alignItems: "center"
            }}
          >
            <p>Click</p>&nbsp;
            <SettingsIcon />
            &nbsp;
            <p>to see all videos in the library</p>
          </div> */}
        </div>
      ) : (
        <div className={classes.container}>
          <Masonry
            className={""} // default ''
            elementType={"div"} // default 'div'
            options={masonryOptions} // default {}
            disableImagesLoaded={false} // default false
            updateOnEachImageLoad={false} // default false and works only if disableImagesLoaded is false
            imagesLoadedOptions={{}} // default {}
          >
            {renderResults.map((data, index) => {
              const [src, distance] = data;
              return (
                <div
                  className={clsx(
                    classes.imgWrapper,
                    index === 0 ? "best" : ""
                  )}
                  key={src}
                >
                  <GifPlayer
                    gif={src}
                    autoplay
                    onLoad={() => {
                      render_nums_ref.current += 1;
                      _loadNewImages();
                    }}
                  />
                  <div className={classes.info}>
                    <p>{(distance || 0.0).toFixed(5)}</p>
                  </div>
                </div>
              );
            })}
          </Masonry>
        </div>
      )}
    </div>
  );
};

export default Results;
