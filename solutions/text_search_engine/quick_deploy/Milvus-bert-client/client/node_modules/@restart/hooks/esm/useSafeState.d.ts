import { Dispatch, SetStateAction } from 'react';
import { AsyncSetState } from './useStateAsync';
declare type StateSetter<TState> = Dispatch<SetStateAction<TState>>;
/**
 * `useSafeState` takes the return value of a `useState` hook and wraps the
 * setter to prevent updates onces the component has unmounted. Can used
 * with `useMergeState` and `useStateAsync` as well
 *
 * @param state The return value of a useStateHook
 *
 * ```ts
 * const [show, setShow] = useSafeState(useState(true));
 * ```
 */
declare function useSafeState<TState>(state: [TState, AsyncSetState<TState>]): [TState, (stateUpdate: React.SetStateAction<TState>) => Promise<void>];
declare function useSafeState<TState>(state: [TState, StateSetter<TState>]): [TState, StateSetter<TState>];
export default useSafeState;
