// @flow

import type { Location, Match, RouterHistory } from 'react-router-dom';

export type ViewType = { [key: string]: any };
export type ViewsType = Array<ViewType>;

export type PropsWithStyles = {
  getStyles: (string, any) => {},
};

export type RouterProps = {
  history: RouterHistory,
  location: Location,
  match: Match,
};
