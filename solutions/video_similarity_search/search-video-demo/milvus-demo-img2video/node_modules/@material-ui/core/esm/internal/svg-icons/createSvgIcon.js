import _extends from "@babel/runtime/helpers/esm/extends";
import React from 'react';
import SvgIcon from '../../SvgIcon';
export default function createSvgIcon(path, displayName) {
  var Component = React.memo(React.forwardRef(function (props, ref) {
    return React.createElement(SvgIcon, _extends({}, props, {
      ref: ref
    }), path);
  }));

  if (process.env.NODE_ENV !== 'production') {
    Component.displayName = "".concat(displayName, "Icon");
  }

  Component.muiName = SvgIcon.muiName;
  return Component;
}