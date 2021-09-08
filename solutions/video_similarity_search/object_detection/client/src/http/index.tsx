import axiosInstance from "./axios";
import { IGetImagePath, IImageUpload } from "../types";
import { API, TABLE_NAME } from "./constants";

const errorParser = (e: any) => {
  console.log(e);
};

export const getImagePath = async (params: IGetImagePath): Promise<any> => {
  return await axiosInstance.get(API.GET_IMAGE, { params }).catch(errorParser);
};

export const progress = async (): Promise<any> => {
  return await axiosInstance.get(API.PROGRESS).catch(errorParser);
};

export const imgUpload = async (filePath: string): Promise<any> => {
  return await axiosInstance
    .post(API.IMG_UPLOAD, {
      Table: TABLE_NAME,
      File: filePath,
    })
    .catch(errorParser);
};

export const getCount = async (): Promise<any> => {
  return await axiosInstance
    .post(API.COUNT, { table_name: TABLE_NAME })
    .catch(errorParser);
};

export const dropTable = async (): Promise<any> => {
  return await axiosInstance
    .post(API.DROP, { table_name: TABLE_NAME })
    .catch(errorParser);
};

export const videoSearch = async (): Promise<any> => {
  return await axiosInstance
    .post(API.SEARCH, { table_name: TABLE_NAME })
    .catch(errorParser);
};
