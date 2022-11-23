declare global {
  interface Window {
    _env_: any;
  }
}
// let endpoint = "http://192.168.1.58:8001";
// let endpoint = `http://localhost:9001`;
let endpoint = "http://172.16.20.7:5000";
if (window._env_ && window._env_.API_URL) {
  endpoint = window._env_.API_URL;
}

export const VIDEO = `${endpoint}/data`;
export const SEARCH = `${endpoint}/video/search`;
export const QUERY_STATUS = `${endpoint}/progress`;
export const LOAD = `${endpoint}/video/load`;
export const COUNT = `${endpoint}/video/count`;
export const DROP = `${endpoint}/video/drop`;
// export const VIDEO = `${endpoint}/v1/video`;
// export const SEARCH = `${endpoint}/v1/search`;
// export const QUERY_STATUS = `${endpoint}/v1/status`;
