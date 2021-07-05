"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement(_react.default.Fragment, null, _react.default.createElement("path", {
  d: "M7 17v5h10v-5H7z"
}), _react.default.createElement("path", {
  fillOpacity: ".3",
  d: "M17 4h-3V2h-4v2H7v13h10V4z"
})), 'Battery20Sharp');

exports.default = _default;