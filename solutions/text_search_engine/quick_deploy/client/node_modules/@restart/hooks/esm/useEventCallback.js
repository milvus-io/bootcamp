import { useCallback } from 'react';
import useCommittedRef from './useCommittedRef';
export default function useEventCallback(fn) {
  var ref = useCommittedRef(fn);
  return useCallback(function () {
    return ref.current && ref.current.apply(ref, arguments);
  }, [ref]);
}