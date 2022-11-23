import axios from "axios";
let API_URL = "http://0.0.0.0:0000/";

console.log(window._env_);
if (window._env_ && window._env_.API_URL) {
  API_URL = window._env_.API_URL;
}

const axiosInstance = axios.create({
  baseURL: API_URL,
  timeout: 50000,
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
    return Promise.resolve(response);
  },
  (error) => {
    return Promise.reject(error);
  }
);
export default axiosInstance;
