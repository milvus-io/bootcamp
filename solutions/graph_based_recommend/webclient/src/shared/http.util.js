import { BASE_URL } from './constants';

export const sendRequest = async (method, path, queryParam, bodyParam) => {
  const param = {
    method,
    headers: {
      'Content-Type': 'application/json',
    },
  };
  let requestUrl = new URL(`${BASE_URL}/${path}`);
  if (queryParam) {
    Object.keys(queryParam).forEach((key) =>
      requestUrl.searchParams.append(key, queryParam[key])
    );
  }
  if (bodyParam) {
    param.body = JSON.stringify(bodyParam);
  }

  const response = await fetch(requestUrl, param);
  const [data, statusCode] = await response.json();
  return {
    data,
    statusCode,
  };
};
