/* Routes */
export const HOME_ROUTE = '/';
export const RECOMMEND_ROUTE = '/recommend';

/* Api */
export const BASE_URL =
  ((window as any)._env_ && (window as any)._env_.API_URL) ||
  'http://192.168.1.85:8008';
