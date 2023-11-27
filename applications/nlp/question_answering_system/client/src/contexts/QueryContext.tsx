import React, { FC, createContext, ReactNode, useState } from "react";
import Snackbar from "@material-ui/core/Snackbar";
import axios from "axios";
import * as URL from "../utils/Endpoints";

const axiosInstance = axios.create();

export const queryContext = createContext<any>({});
const Provider = queryContext.Provider;

type SnackBar = {
  open: boolean;
  message: string;
};

const QueryProvider: FC<{ children: ReactNode }> = ({ children }) => {
  const [snackBarConfig, setSnackBarConfig] = useState<SnackBar>({
    open: false,
    message: "",
  });

  const errorParser = (e: any) => {
    console.log(e);
  };
  const process = async (params: any) => {
    const url = URL.Processing;
    return await axiosInstance.get(url, params).catch(errorParser);
  };
  const count = async (params: any) => {
    const url = URL.Count;
    return await axiosInstance.post(url, params).catch(errorParser);
  };
  const train = async (params: any) => {
    const url = URL.Train;
    return await axiosInstance.post(url, params).catch(errorParser);
  };

  const clearAll = async () => {
    const url = URL.ClearAll;
    return await axiosInstance.post(url).catch(errorParser);
  };

  const load = async (params: { file: FormData }) => {
    const url = URL.LOAD;
    return await axiosInstance.post(url, params).catch(errorParser);
  };

  const search = async (params: { question: string }) => {
    const url = URL.SEARCH;
    return await axiosInstance.get(url, { params }).catch(errorParser);
  };

  const getAnswer = async (params: { question: string }) => {
    const url = URL.ANSWER;
    return await axiosInstance.get(url, { params }).catch(errorParser);
  };

  const dropTable = async () => {
    const url = URL.DROP;
    return await axiosInstance.post(url).catch(errorParser);
  };

  const openSnackBar = (message: string) =>
    setSnackBarConfig({
      open: true,
      message,
    });

  const closeSnackBar = () =>
    setSnackBarConfig({
      open: false,
      message: "",
    });

  return (
    <Provider
      value={{
        process,
        count,
        search,
        clearAll,
        train,
        load,
        dropTable,
        getAnswer,
        openSnackBar,
        closeSnackBar,
      }}
    >
      {children}
      <Snackbar
        anchorOrigin={{ vertical: "top", horizontal: "right" }}
        {...snackBarConfig}
        onClose={closeSnackBar}
        key={"top" + "right"}
      />
    </Provider>
  );
};

export default QueryProvider;
