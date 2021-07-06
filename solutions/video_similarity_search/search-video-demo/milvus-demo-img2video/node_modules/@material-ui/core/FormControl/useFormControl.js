"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = useFormControl;

var _react = _interopRequireDefault(require("react"));

var _FormControlContext = _interopRequireDefault(require("./FormControlContext"));

function useFormControl() {
  return _react.default.useContext(_FormControlContext.default);
}