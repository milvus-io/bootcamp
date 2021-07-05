"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement("path", {
  d: "M14 8.83L17.17 12 14 15.17V14H6v-4h8V8.83M12 4v4H4v8h8v4l8-8-8-8z"
}), 'ForwardOutlined');

exports.default = _default;