"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement("path", {
  d: "M19 11h-8v6h8v-6zm4 10V3H1v18h22zm-2-1.98H3V4.97h18v14.05z"
}), 'PictureInPictureAltSharp');

exports.default = _default;