import useUpdatedRef from './useUpdatedRef';
import { useEffect } from 'react';
/**
 * Attach a callback that fires when a component unmounts
 *
 * @param fn Handler to run when the component unmounts
 * @category effects
 */

export default function useWillUnmount(fn) {
  var onUnmount = useUpdatedRef(fn);
  useEffect(function () {
    return function () {
      return onUnmount.current();
    };
  }, []);
}