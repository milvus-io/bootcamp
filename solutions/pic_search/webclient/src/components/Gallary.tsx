import React, { useState } from "react";
import { createStyles, makeStyles } from "@material-ui/core/styles";
import { baseColor } from "../utils/color";
import useMediaQuery from "@material-ui/core/useMediaQuery";

const placeHolderImges = new Array(20).fill({});
const Gallary = (props: any) => {
  const isMobile = !useMediaQuery("(min-width:1000px)");
  const useStyles = makeStyles((theme) =>
    createStyles({
      root: {
        display: isMobile ? "block:" : "flex",
        justifyContent: "flex-start",
        alignItems: "center",
        flexWrap: "wrap",
      },
      title: {
        marginTop: "20px",
        fontSize: "20px",
      },
      imageContainer: {
        position: "relative",
        flex: isMobile ? 1 : "0 0 21%",
        margin: isMobile ? 0 : "0 15px 15px 0",
        paddingTop: isMobile ? "100%" : "20%",
        border: "solid 1px #60606F",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        fontSize: "50px",
        width: isMobile ? "100%" : "auto",
        height: isMobile ? "0" : "auto",
      },
      child: {
        width: "100%",
        maxHeight: "100%",
        position: "absolute",
        top: `0`,
        bottom: `0`,
        margin: "auto",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        color: "#60606F",
        fontSize: "8vw",
      },
      distant: {
        fontSize: "1vw",
        zIndex: 1000,
        color: "#fafafa",
        backgroundColor: "#000",
        position: "absolute",
        bottom: "0px",
        left: "0px",
        width: "100%",
        height: "24px",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        overflowX: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
      },
    })
  );
  const { images = [], onClick } = props;
  const classes = useStyles({});
  const [hovered, setHovered]: any = useState();
  const onMouseOver = (e: any) => {
    const src = e.currentTarget.dataset.src;
    setHovered(src);
  };
  const onMouseLeave = () => {
    setHovered(null);
  };
  return (
    <div className={classes.root}>
      {images.length > 0
        ? images.map((img: any, index: number) => {
            const { src, distance } = img;
            const isHovered = hovered === src;
            return (
              <div
                className={classes.imageContainer}
                onClick={() => {
                  onClick(index);
                }}
                onMouseOver={onMouseOver}
                onMouseLeave={onMouseLeave}
                key={src}
                data-src={src}
                style={{
                  border: `solid 2px ${isHovered ? baseColor : "transparent"}`,
                }}
              >
                <img src={src} className={classes.child} alt="" />
                {isHovered && (
                  <p className={classes.distant}>{distance.toFixed(6)}</p>
                )}
              </div>
            );
          })
        : placeHolderImges.map((item: any, index: number) => (
            <div className={classes.imageContainer} key={index}>
              <p className={classes.child}>{index < 12 ? index + 1 : ""}</p>
            </div>
          ))}
    </div>
  );
};

export default Gallary;
