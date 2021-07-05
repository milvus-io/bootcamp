"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = useTheme;

var _react = _interopRequireDefault(require("react"));

var _ThemeContext = _interopRequireDefault(require("./ThemeContext"));

function useTheme() {
  return _react.default.useContext(_ThemeContext.default);
}