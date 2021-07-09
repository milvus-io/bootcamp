import axios from "axios";

let API_URL = "http://192.168.1.85:8002/";

if (window._env_ && window._env_.API_URL) {
  API_URL = window._env_.API_URL;
}

const axiosInstance = axios.create({
  baseURL: API_URL,
  timeout: 20000,
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
  (res) => {
    const { data, code } = res;
    switch (code) {
      case 200:
        return Promise.resolve(data);
      default:
        return Promise.resolve({
          code,
          msg: "Request Error!",
        });
    }
  },
  (error) => {
    return Promise.reject(error);
  }
);

export default axiosInstance;
