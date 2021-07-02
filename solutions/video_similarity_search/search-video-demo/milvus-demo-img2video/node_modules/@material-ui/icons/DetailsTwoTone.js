"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement(_react.default.Fragment, null, _react.default.createElement("path", {
  d: "M6.38 6L12 16l5.63-10z",
  opacity: ".3"
}), _react.default.createElement("path", {
  d: "M3 4l9 16 9-16H3zm3.38 2h11.25L12 16 6.38 6z"
})), 'DetailsTwoTone');

exports.default = _default;