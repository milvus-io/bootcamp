"use strict";

exports.__esModule = true;
exports.default = usePrevious;

var _react = require("react");

/**
 * Store the last of some value. Tracked via a `Ref` only updating it
 * after the component renders.
 *
 * Helpful if you need to compare a prop value to it's previous value during render.
 *
 * ```ts
 * function Component(props) {
 *   const lastProps = usePrevious(props)
 *
 *   if (lastProps.foo !== props.foo)
 *     resetValueFromProps(props.foo)
 * }
 * ```
 *
 * @param value the value to track
 */
function usePrevious(value) {
  var ref = (0, _react.useRef)(null);
  (0, _react.useEffect)(function () {
    ref.current = value;
  });
  return ref.current;
}