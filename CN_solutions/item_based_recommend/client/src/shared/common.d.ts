export type Method = 'GET' | 'POST' | 'PUT' | 'DELETE' | undefined;

export interface IParam {
  method: Method;
  path: string;
  queryParam?: {
    [key: string]: any;
  };
  bodyParam?: {
    [k: string]: any;
  };
}

export interface IRequestParam {
  method: Method;
  headers: HeadersInit;
  body?: any;
}

export interface IResponse {
  statusCode?: number;
  data: any;
}
