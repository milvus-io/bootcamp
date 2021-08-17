"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _createSvgIcon = _interopRequireDefault(require("./utils/createSvgIcon"));

var _default = (0, _createSvgIcon.default)(_react.default.createElement(_react.default.Fragment, null, _react.default.createElement("path", {
  d: "M4 4v13.17l.59-.59.58-.58H20V4H4zm9 11h-2v-2h2v2zm0-4h-2V5h2v6z",
  opacity: ".3"
}), _react.default.createElement("path", {
  d: "M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H5.17l-.59.59-.58.58V4h16v12zM11 5h2v6h-2zm0 8h2v2h-2z"
})), 'AnnouncementTwoTone');

exports.default = _default;