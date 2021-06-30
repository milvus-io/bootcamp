"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement(_react.default.Fragment, null, _react.default.createElement("path", {
  d: "M6 13h13v3H6zm0-5h13v3H6z",
  opacity: ".3"
}), _react.default.createElement("path", {
  d: "M4 6v12h17V6H4zm15 10H6v-3h13v3zm0-5H6V8h13v3z"
})), 'ViewStreamTwoTone');

exports.default = _default;