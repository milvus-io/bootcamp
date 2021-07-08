"use strict";

exports.__esModule = true;
exports.default = useEventCallback;

var _react = require("react");

var _useCommittedRef = _interopRequireDefault(require("./useCommittedRef"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function useEventCallback(fn) {
  var ref = (0, _useCommittedRef.default)(fn);
  return (0, _react.useCallback)(function () {
    return ref.current && ref.current.apply(ref, arguments);
  }, [ref]);
}