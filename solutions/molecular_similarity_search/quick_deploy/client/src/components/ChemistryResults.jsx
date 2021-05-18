import React, { useState, useCallback } from "react";
import { makeStyles } from "@material-ui/core/styles";
import Carousel, { Modal, ModalGateway } from "react-images";
import ResultHeader from "./ChemistryResultHeader";
import Result from "./ChemistryResult";

const useStyles = makeStyles({
  root: {
    flexGrow: 1,
    overflowX: "hidden",
    overflowY: "auto",
    padding: "80px 60px",
    display: "flex",
    flexDirection: "column",
    backgroundColor: "#28292E"
  },
  title: {
    margin: "20px 0px 10px",
    fontSize: "20px",
    color: "#F5F5F5"
  },
  subTitle: {
    fontSize: "15px",
    color: "#F1F1F1",
    marginBottom: "１0px !important"
  }
});

const SearchResults = props => {
  const classes = useStyles({});
  const { results = [] } = props;
  const [currentImage, setCurrentImage] = useState(0);
  const [viewerIsOpen, setViewerIsOpen] = useState(false);
  const images = Object.keys(results);
  const datas = images.map(image => {
    return {
      src: image,
      Molecular: results[image][0],
      Distance: results[image][1]
    };
  });
  const _openLightbox = useCallback(index => {
    setCurrentImage(index);
    setViewerIsOpen(true);
  }, []);

  const _closeLightbox = () => {
    setCurrentImage(0);
    setViewerIsOpen(false);
  };

  return (
    <div className={classes.root}>
      <div className={classes.title}>
        <h3 className={classes.title}>Search Results</h3>
      </div>
      <ResultHeader
        title={"Structure Picture"}
        Molecular={"Molecular"}
        Distance={"Distance"}
        style={{ backgroundColor: "#000" }}
      />
      {datas.length === 0 && <div></div>}
      <>
        {datas.map((data, index) => {
          return (
            <Result
              {...data}
              key={index}
              onClick={() => _openLightbox(index)}
              style={{ backgroundColor: index % 2 ? "#323338" : "#28292e" }}
            />
          );
        })}
      </>
      <ModalGateway>
        {viewerIsOpen ? (
          <Modal onClose={_closeLightbox}>
            <Carousel
              currentIndex={currentImage}
              views={datas.map(x => ({
                ...x,
                srcset: x.srcSet,
                caption: x.title
              }))}
            />
          </Modal>
        ) : null}
      </ModalGateway>
    </div>
  );
};

export default SearchResults;
