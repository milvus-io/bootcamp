"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement(_react.default.Fragment, null, _react.default.createElement("path", {
  d: "M12 2l-5.5 9h11z"
}), _react.default.createElement("circle", {
  cx: "17.5",
  cy: "17.5",
  r: "4.5"
}), _react.default.createElement("path", {
  d: "M3 13.5h8v8H3z"
})), 'CategorySharp');

exports.default = _default;