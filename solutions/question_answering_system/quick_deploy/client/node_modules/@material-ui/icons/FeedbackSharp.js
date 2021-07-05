"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement("path", {
  d: "M22 2H2.01L2 22l4-4h16V2zm-9 12h-2v-2h2v2zm0-4h-2V6h2v4z"
}), 'FeedbackSharp');

exports.default = _default;