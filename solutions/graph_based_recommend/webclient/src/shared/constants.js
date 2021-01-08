/* Routes */
export const HOME_ROUTE = '/';
export const RECOMMEND_ROUTE = '/recommend';
export const DETAIL_ROUTE = '/detail';

/* Api */
export const BASE_URL =
  (window._env_ && window._env_.API_URL) || 'http://192.168.1.85:8000';
