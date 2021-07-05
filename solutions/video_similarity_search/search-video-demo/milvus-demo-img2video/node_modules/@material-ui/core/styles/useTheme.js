"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = useTheme;

var _styles = require("@material-ui/styles");

var _defaultTheme = _interopRequireDefault(require("./defaultTheme"));

function useTheme() {
  return (0, _styles.useTheme)() || _defaultTheme.default;
}