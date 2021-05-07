import { IParam, IRequestParam, IResponse } from './common';
import { BASE_URL } from './constansts';

export const sendRequest = async (param: IParam): Promise<IResponse> => {
  const { method, path, queryParam, bodyParam } = param;
  const requestParam: IRequestParam = {
    method,
    headers: {
      'Content-Type': 'application/json',
    },
  };

  let url = `${BASE_URL}/${path}`;
  if (queryParam) {
    const params = new URLSearchParams(queryParam);
    url += `?${params.toString()}`;
  }

  if (bodyParam) {
    requestParam.body = JSON.stringify(bodyParam);
  }

  const response = await fetch(url, requestParam);
  const data = await response.json();
  return { data };
};
