"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

exports.__esModule = true;
exports["default"] = useWaitForDOMRef;

var _ownerDocument = _interopRequireDefault(require("dom-helpers/ownerDocument"));

var _react = require("react");

var resolveRef = function resolveRef(ref) {
  if (typeof document === 'undefined') return undefined;
  if (ref == null) return (0, _ownerDocument["default"])().body;
  if (typeof ref === 'function') ref = ref();
  if (ref && ref.current) ref = ref.current;
  if (ref && ref.nodeType) return ref;
  return null;
};

function useWaitForDOMRef(ref, onResolved) {
  var _useState = (0, _react.useState)(function () {
    return resolveRef(ref);
  }),
      resolvedRef = _useState[0],
      setRef = _useState[1];

  if (!resolvedRef) {
    var earlyRef = resolveRef(ref);
    if (earlyRef) setRef(earlyRef);
  }

  (0, _react.useEffect)(function () {
    if (onResolved && resolvedRef) {
      onResolved(resolvedRef);
    }
  }, [onResolved, resolvedRef]);
  (0, _react.useEffect)(function () {
    var nextRef = resolveRef(ref);

    if (nextRef !== resolvedRef) {
      setRef(nextRef);
    }
  }, [ref, resolvedRef]);
  return resolvedRef;
}

module.exports = exports.default;