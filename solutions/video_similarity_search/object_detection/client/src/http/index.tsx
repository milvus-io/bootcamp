import axios from "axios";
import { IGetImagePath } from "../types";

const axiosInstance = axios.create({
  baseURL: "",
  timeout: 20000,
  headers: {
    "Content-Type": "application/json",
  },
});

axiosInstance.interceptors.request.use(
  (config) => {
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

axiosInstance.interceptors.response.use(
  (response) => {
    return new Promise(response.data);
  },
  (error) => {
    return Promise.reject(error);
  }
);

const errorParser = (e: any) => {
  console.log(e);
};

export const gteImagePath = async (
  url: string,
  params: IGetImagePath
): Promise<any> => {
  return await axiosInstance.post(url, params).catch(errorParser);
};
