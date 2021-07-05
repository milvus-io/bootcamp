"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement("path", {
  transform: "scale(0.5, 0.5)",
  d: "M42 20c-2.21 0-4 1.79-4 4v6H10v-6c0-2.21-1.79-4-4-4s-4 1.79-4 4v10c0 2.2 1.8 4 4 4h36c2.2 0 4-1.8 4-4V24c0-2.21-1.79-4-4-4zm-6-10H12c-2.2 0-4 1.8-4 4v4.31c2.32.83 4 3.03 4 5.63V28h24v-4.06c0-2.6 1.68-4.8 4-5.63V14c0-2.2-1.8-4-4-4z"
}), 'Weekend');

exports.default = _default;