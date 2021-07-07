import { useCallback, useEffect, useRef, useState } from 'react';

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
  var _useState = useState(initialState),
      state = _useState[0],
      setState = _useState[1];

  var resolvers = useRef([]);
  useEffect(function () {
    resolvers.current.forEach(function (resolve) {
      return resolve(state);
    });
    resolvers.current.length = 0;
  }, [state]);
  var setStateAsync = useCallback(function (update) {
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

export default useStateAsync;