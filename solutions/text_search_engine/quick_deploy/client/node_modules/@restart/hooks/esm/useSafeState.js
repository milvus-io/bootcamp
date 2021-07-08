import { useCallback } from 'react';
import useMounted from './useMounted';

function useSafeState(state) {
  var isMounted = useMounted();
  return [state[0], useCallback(function (nextState) {
    if (!isMounted()) return;
    return state[1](nextState);
  }, [isMounted, state[1]])];
}

export default useSafeState;