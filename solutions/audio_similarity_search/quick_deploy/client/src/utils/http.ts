import axiosInstance from "./axios";

const LOAD = "audio/load";
const DROP = "audio/drop";
const SEARCH = "audio/search";
const COUNT = "audio/count";
const PROGRESS = "progress";
const TABLE_NAME = "audio_table";

const uploadSet = async (params: any) => {
  return await axiosInstance.post(LOAD, params);
};

const dropSet = async (params: any) => {
  return await axiosInstance.post(DROP, params);
};

const search = async (params: any) => {
  return await axiosInstance.post(SEARCH, params);
};

const count = async (param: any) => {
  return await axiosInstance.get(COUNT, param);
};

export { uploadSet, dropSet, search, count, TABLE_NAME };
