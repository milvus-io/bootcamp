import React from 'react';
export declare type AsyncSetState<TState> = (stateUpdate: React.SetStateAction<TState>) => Promise<TState>;
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
declare function useStateAsync<TState>(initialState: TState | (() => TState)): [TState, AsyncSetState<TState>];
export default useStateAsync;
