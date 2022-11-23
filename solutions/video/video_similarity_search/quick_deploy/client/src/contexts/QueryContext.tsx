import React, { FC, createContext, ReactNode, useState } from "react";
import { makeStyles } from "@material-ui/core/styles";
import axios from "axios";
import ErrorIcon from "@material-ui/icons/ErrorOutline";
import Snackbar from "@material-ui/core/Snackbar";
import * as URL from "../utils/Endpoints";

const axiosInstance = axios.create();
export const queryContext = createContext<any>({});
const Provider = queryContext.Provider;

const useStyles = makeStyles({
  root: {
    marginTop: "20%",
  },
  color: {
    color: "red",
    marginRight: "10px",
  },
});

// const isMock = false;
const QueryProvider: FC<{ children: ReactNode }> = ({ children }) => {
  const classes = useStyles();
  // current page : search | library
  const [page, setPage] = useState<string>("search");
  const [navTitle, setNavTitle]: any = useState("Waiting For Search");
  // page status : upload-img | search | show-search | upload-library| show-library | fail-library
  const [pageStatus, setPageStatus] = useState<string>("show-search");
  // note status
  const [noteStatus, setNoteStatus] = useState<any>({
    show: false,
    content: "",
  });
  // searchParams
  const [searchParams, setSearchParams]: any = useState({
    history: [],
    curr: {
      file: "",
      data: "",
      id: "",
    },
  });
  const showNote = (content: string) => {
    setNoteStatus({ show: true, content });
  };

  const hideNote = () => setNoteStatus({ show: false, content: "" });

  const errorParser = (e: any) => {
    console.log(e);
  };
  const search = async (params: any) => {
    const url = URL.SEARCH;
    const bodyFormData = new FormData();
    bodyFormData.set("image", params);
    return await axiosInstance.post(url, bodyFormData).catch(errorParser);
  };
  const upload = async (params: any) => {
    const url = URL.VIDEO;
    var formData = new FormData();
    formData.set("file", params);
    return await axiosInstance.post(url, formData).catch(errorParser);
  };
  const queryStatus = async () => {
    const url = URL.QUERY_STATUS;
    return await axiosInstance.get(url).catch(errorParser);
  };
  const queryLibrary = async (params: any) => {
    const url = URL.VIDEO;
    params = {
      Reverse: true,
      PageNum: params.page,
      PerPageCount: 30,
    };
    return await axiosInstance.get(url, { params }).catch(errorParser);
  };
  const delVideo = async (_params: any) => {
    const url = URL.VIDEO;
    const params = { ID: _params };
    return await axiosInstance.delete(url, { params }).catch(errorParser);
  };

  const videosLoad = async (_params: any) => {
    const url = URL.LOAD;
    return await axiosInstance.post(url, _params).catch(errorParser);
  };

  const videosDrop = async () => {
    const url = URL.DROP;
    return await axiosInstance.post(url).catch(errorParser);
  };
  const getCount = async () => {
    const url = URL.COUNT;
    return await axiosInstance.post(url).catch(errorParser);
  };
  return (
    <Provider
      value={{
        navTitle,
        setNavTitle,
        // querys
        search,
        upload,
        queryStatus,
        queryLibrary,
        delVideo,
        videosLoad,
        videosDrop,
        getCount,
        // notes
        showNote,
        hideNote,
        // pages
        page,
        setPage,
        pageStatus,
        setPageStatus,
        noteStatus,
        setNoteStatus,

        searchParams,
        setSearchParams,
      }}
    >
      {children}
      <Snackbar
        classes={{ root: classes.root }}
        open={noteStatus.show}
        anchorOrigin={{
          vertical: "top",
          horizontal: "center",
        }}
        autoHideDuration={6000}
        onClose={() => hideNote()}
        message={
          <div
            style={{
              display: "flex",
              alignItems: "center",
              color: "red",
            }}
          >
            <ErrorIcon classes={{ root: classes.color }} />
            <span>{noteStatus.content || ""}</span>
          </div>
        }
      ></Snackbar>
    </Provider>
  );
};

export default QueryProvider;
