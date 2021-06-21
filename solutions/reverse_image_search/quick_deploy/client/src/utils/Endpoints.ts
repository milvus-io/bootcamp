declare global {
    interface Window {
        _env_: any;
    }
}

let endpoint = `http://172.16.20.10:5000`;
if (window._env_ && window._env_.API_URL) {
    endpoint = window._env_.API_URL;
}

export const Train = `${endpoint}/img/load`;
export const Processing = `${endpoint}/progress`;
export const Count = `${endpoint}/img/count`;
export const ClearAll = `${endpoint}/img/drop`;
export const Search = `${endpoint}/img/search`;
export const GetImageUrl = `${endpoint}/data`;
