"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

exports.__esModule = true;
exports["default"] = void 0;

var _propTypes = _interopRequireDefault(require("prop-types"));

var _reactDom = _interopRequireDefault(require("react-dom"));

var _useWaitForDOMRef = _interopRequireDefault(require("./utils/useWaitForDOMRef"));

var propTypes = {
  /**
   * A DOM element, Ref to an element, or function that returns either. The `container` will have the Portal children
   * appended to it.
   */
  container: _propTypes["default"].any,
  onRendered: _propTypes["default"].func
};
/**
 * The `<Portal/>` component renders its children into a new "subtree" outside of current component hierarchy.
 * You can think of it as a declarative `appendChild()`, or jQuery's `$.fn.appendTo()`.
 * The children of `<Portal/>` component will be appended to the `container` specified.
 *
 */

var Portal = function Portal(_ref) {
  var container = _ref.container,
      children = _ref.children,
      onRendered = _ref.onRendered;
  var resolvedContainer = (0, _useWaitForDOMRef["default"])(container, onRendered);
  return resolvedContainer ? _reactDom["default"].createPortal(children, resolvedContainer) : null;
};

Portal.displayName = 'Portal';
Portal.propTypes = propTypes;
/**
 * @component
 */

var _default = Portal;
exports["default"] = _default;
module.exports = exports.default;