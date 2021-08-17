"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement("path", {
  d: "M21 0H3v24h18V0zm-7 22h-4v-1h4v1zm5.25-3H4.75V3h14.5v16z"
}), 'TabletAndroidSharp');

exports.default = _default;