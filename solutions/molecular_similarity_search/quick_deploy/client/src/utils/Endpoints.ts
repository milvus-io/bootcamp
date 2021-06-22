declare global {
  interface Window {
    _env_: any;
  }
}
let endpoint = `http://172.16.20.7:5001`;
if (window._env_ && window._env_.API_URL) {
  endpoint = window._env_.API_URL;
}

export const LOAD = `${endpoint}/data/load`;
export const PROCESSING = `${endpoint}/progress`;
export const COUNT = `${endpoint}/data/count`;
export const CLEAR_ALL = `${endpoint}/data/drop`;
export const SEARCH = `${endpoint}/data/search`;

export const PATH_IMAGE = `${endpoint}/data/workspace/apptec/demo/test_100.smi`;
