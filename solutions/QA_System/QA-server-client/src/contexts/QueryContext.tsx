import React, { FC, createContext, ReactNode } from "react";
import axios from "axios";
import * as URL from "../utils/Endpoints";

const axiosInstance = axios.create();

export const queryContext = createContext<any>({});
const Provider = queryContext.Provider;

const QueryProvider: FC<{ children: ReactNode }> = ({ children }) => {
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

  const search = async (params: any) => {
    const url = URL.Search;
    return await axiosInstance.post(url, params).catch(errorParser);
  };
  const clearAll = async () => {
    const url = URL.ClearAll;
    return await axiosInstance.post(url).catch(errorParser);
  };

  return (
    <Provider
      value={{
        process,
        count,
        search,
        clearAll,
        train
      }}
    >
      {children}
    </Provider>
  );
};

export default QueryProvider;
