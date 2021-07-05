"use strict";

exports.__esModule = true;
exports.mergeRefs = mergeRefs;
exports.default = void 0;

var _react = require("react");

var toFnRef = function toFnRef(ref) {
  return !ref || typeof ref === 'function' ? ref : function (value) {
    ref.current = value;
  };
};

function mergeRefs(refA, refB) {
  var a = toFnRef(refA);
  var b = toFnRef(refB);
  return function (value) {
    if (a) a(value);
    if (b) b(value);
  };
}
/**
 * Create and returns a single callback ref composed from two other Refs.
 *
 * ```tsx
 * const Button = React.forwardRef((props, ref) => {
 *   const [element, attachRef] = useCallbackRef<HTMLButtonElement>();
 *   const mergedRef = useMergedRefs(ref, attachRef);
 *
 *   return <button ref={mergedRef} {...props}/>
 * })
 * ```
 *
 * @param refA A Callback or mutable Ref
 * @param refB A Callback or mutable Ref
 * @category refs
 */


function useMergedRefs(refA, refB) {
  return (0, _react.useMemo)(function () {
    return mergeRefs(refA, refB);
  }, [refA, refB]);
}

var _default = useMergedRefs;
exports.default = _default;