"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement("path", {
  d: "M19 3H5.01L5 21l7-3 7 3V3zm-2 15l-5-2.18L7 18V5h10v13z"
}), 'TurnedInNotSharp');

exports.default = _default;