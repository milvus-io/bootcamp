import { useEffect, useRef } from 'react';
/**
 * Store the last of some value. Tracked via a `Ref` only updating it
 * after the component renders.
 *
 * Helpful if you need to compare a prop value to it's previous value during render.
 *
 * ```ts
 * function Component(props) {
 *   const lastProps = usePrevious(props)
 *
 *   if (lastProps.foo !== props.foo)
 *     resetValueFromProps(props.foo)
 * }
 * ```
 *
 * @param value the value to track
 */

export default function usePrevious(value) {
  var ref = useRef(null);
  useEffect(function () {
    ref.current = value;
  });
  return ref.current;
}