"use strict";

exports.__esModule = true;
exports.default = useMergeStateFromProps;

var _useMergeState2 = _interopRequireDefault(require("./useMergeState"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function useMergeStateFromProps(props, gDSFP, initialState) {
  var _useMergeState = (0, _useMergeState2.default)(initialState),
      state = _useMergeState[0],
      setState = _useMergeState[1];

  var nextState = gDSFP(props, state);
  if (nextState !== null) setState(nextState);
  return [state, setState];
}