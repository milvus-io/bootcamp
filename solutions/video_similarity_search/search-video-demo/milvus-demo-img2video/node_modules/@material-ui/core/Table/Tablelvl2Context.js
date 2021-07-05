"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

/**
 * @ignore - internal component.
 */
var Tablelvl2Context = _react.default.createContext();

if (process.env.NODE_ENV !== 'production') {
  Tablelvl2Context.displayName = 'Tablelvl2Context';
}

var _default = Tablelvl2Context;
exports.default = _default;