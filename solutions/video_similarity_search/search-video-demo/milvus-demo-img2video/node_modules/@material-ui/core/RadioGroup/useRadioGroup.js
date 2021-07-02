"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = useRadioGroup;

var _react = _interopRequireDefault(require("react"));

var _RadioGroupContext = _interopRequireDefault(require("./RadioGroupContext"));

function useRadioGroup() {
  return _react.default.useContext(_RadioGroupContext.default);
}