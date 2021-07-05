import { DependencyList } from 'react';
/**
 * Identical to `useMemo` _except_ that it provides a semantic guarantee that
 * values will not be invalidated unless the dependencies change. This is unlike
 * the built in `useMemo` which may discard memoized values for performance reasons.
 *
 * @param factory A function that returns a value to be memoized
 * @param deps A dependency array
 */
export default function useStableMemo<T>(factory: () => T, deps?: DependencyList): T;
