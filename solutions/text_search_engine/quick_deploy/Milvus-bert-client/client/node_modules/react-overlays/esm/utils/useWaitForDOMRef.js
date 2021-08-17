import ownerDocument from 'dom-helpers/ownerDocument';
import { useState, useEffect } from 'react';

var resolveRef = function resolveRef(ref) {
  if (typeof document === 'undefined') return undefined;
  if (ref == null) return ownerDocument().body;
  if (typeof ref === 'function') ref = ref();
  if (ref && ref.current) ref = ref.current;
  if (ref && ref.nodeType) return ref;
  return null;
};

export default function useWaitForDOMRef(ref, onResolved) {
  var _useState = useState(function () {
    return resolveRef(ref);
  }),
      resolvedRef = _useState[0],
      setRef = _useState[1];

  if (!resolvedRef) {
    var earlyRef = resolveRef(ref);
    if (earlyRef) setRef(earlyRef);
  }

  useEffect(function () {
    if (onResolved && resolvedRef) {
      onResolved(resolvedRef);
    }
  }, [onResolved, resolvedRef]);
  useEffect(function () {
    var nextRef = resolveRef(ref);

    if (nextRef !== resolvedRef) {
      setRef(nextRef);
    }
  }, [ref, resolvedRef]);
  return resolvedRef;
}