declare global {
  interface Window {
    _env_: any;
  }
}
// let endpoint = `http://40.117.75.127:5005`;
let endpoint = "http://ec2-18-166-215-54.ap-east-1.compute.amazonaws.com:8000/";
if (window._env_ && window._env_.API_URL) {
  endpoint = window._env_.API_URL;
}
export const Train = `${endpoint}/api/v1/train`;
export const Processing = `${endpoint}/api/v1/process`;
export const Count = `${endpoint}/api/v1/count`;
export const ClearAll = `${endpoint}/api/v1/delete`;
export const Search = `${endpoint}/api/v1/search`;

export const PATH_IMAGE = `${endpoint}/home/zilliz_support/workspace/lcl/milvus_demo/web_test/pic1`;
export const PATH_IMAGE2 = `${endpoint}/home/zilliz_support/workspace/lcl/milvus_demo/web_test/pic2`;

export const LOAD = `${endpoint}/qa/load_data`;
export const SEARCH = `${endpoint}/qa/search`;
export const ANSWER = `${endpoint}/qa/answer`;
export const COUNT = `${endpoint}/qa/count`;
export const DROP = `${endpoint}/qa/drop`;
