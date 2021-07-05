"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

/**
 * @ignore - internal component.
 * @type {React.Context<{} | {expanded: boolean, disabled: boolean, toggle: () => void}>}
 */
var ExpansionPanelContext = _react.default.createContext({});

if (process.env.NODE_ENV !== 'production') {
  ExpansionPanelContext.displayName = 'ExpansionPanelContext';
}

var _default = ExpansionPanelContext;
exports.default = _default;