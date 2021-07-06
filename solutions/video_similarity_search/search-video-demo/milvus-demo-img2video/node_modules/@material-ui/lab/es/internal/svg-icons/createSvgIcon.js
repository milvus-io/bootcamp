import _extends from "@babel/runtime/helpers/esm/extends";
import React from 'react';
import SvgIcon from '@material-ui/core/SvgIcon';
export default function createSvgIcon(path, displayName) {
  const Component = React.memo(React.forwardRef((props, ref) => React.createElement(SvgIcon, _extends({
    ref: ref
  }, props), path)));

  if (process.env.NODE_ENV !== 'production') {
    Component.displayName = `${displayName}Icon`;
  }

  Component.muiName = SvgIcon.muiName;
  return Component;
}