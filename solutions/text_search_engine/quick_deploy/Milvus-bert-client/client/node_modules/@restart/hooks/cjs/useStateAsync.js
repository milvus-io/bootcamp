"use strict";

exports.__esModule = true;
exports.default = void 0;

var _react = require("react");

/**
 * A hook that mirrors `useState` in function and API, expect that setState
 * calls return a promise that resolves after the state has been set (in an effect).
 *
 * This is _similar_ to the second callback in classy setState calls, but fires later.
 *
 * ```ts
 * const [counter, setState] = useStateAsync(1);
 *
 * const handleIncrement = async () => {
 *   await setState(2);
 *   doWorkRequiringCurrentState()
 * }
 * ```
 *
 * @param initialState initialize with some state value same as `useState`
 */
function useStateAsync(initialState) {
  var _useState = (0, _react.useState)(initialState),
      state = _useState[0],
      setState = _useState[1];

  var resolvers = (0, _react.useRef)([]);
  (0, _react.useEffect)(function () {
    resolvers.current.forEach(function (resolve) {
      return resolve(state);
    });
    resolvers.current.length = 0;
  }, [state]);
  var setStateAsync = (0, _react.useCallback)(function (update) {
    return new Promise(function (resolve, reject) {
      setState(function (prevState) {
        try {
          var nextState; // ugly instanceof for typescript

          if (update instanceof Function) {
            nextState = update(prevState);
          } else {
            nextState = update;
          } // If state does not change, we must resolve the promise because
          // react won't re-render and effect will not resolve. If there are already
          // resolvers queued, then it should be safe to assume an update will happen


          if (!resolvers.current.length && Object.is(nextState, prevState)) {
            resolve(nextState);
          } else {
            resolvers.current.push(resolve);
          }

          return nextState;
        } catch (e) {
          reject(e);
          throw e;
        }
      });
    });
  }, [setState]);
  return [state, setStateAsync];
}

var _default = useStateAsync;
exports.default = _default;