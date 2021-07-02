"use strict";

exports.__esModule = true;
exports.default = void 0;

var _react = require("react");

var _useMounted = _interopRequireDefault(require("./useMounted"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function useSafeState(state) {
  var isMounted = (0, _useMounted.default)();
  return [state[0], (0, _react.useCallback)(function (nextState) {
    if (!isMounted()) return;
    return state[1](nextState);
  }, [isMounted, state[1]])];
}

var _default = useSafeState;
exports.default = _default;