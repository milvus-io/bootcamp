declare global {
  interface Window {
    _env_: any;
  }
}
let endpoint = `http://192.168.1.58:5000`;
if (window._env_ && window._env_.API_URL) {
  endpoint = window._env_.API_URL;
}

export const LOAD = `${endpoint}/api/v1/load`;
export const PROCESSING = `${endpoint}/api/v1/process`;
export const COUNT = `${endpoint}/api/v1/count`;
export const CLEAR_ALL = `${endpoint}/api/v1/delete`;
export const SEARCH = `${endpoint}/api/v1/search`;

export const PATH_IMAGE = `${endpoint}/data/workspace/apptec/demo/test_100.smi`;