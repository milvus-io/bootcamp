"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.useFormControl = useFormControl;
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

/**
 * @ignore - internal component.
 */
var FormControlContext = _react.default.createContext();

if (process.env.NODE_ENV !== 'production') {
  FormControlContext.displayName = 'FormControlContext';
}

function useFormControl() {
  return _react.default.useContext(FormControlContext);
}

var _default = FormControlContext;
exports.default = _default;