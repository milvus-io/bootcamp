declare type Updater<TState> = (state: TState) => Partial<TState> | null;
/**
 * Updates state, partial updates are merged into existing state values
 */
export declare type MergeStateSetter<TState> = (update: Updater<TState> | Partial<TState> | null) => void;
/**
 * Mimics a React class component's state model, of having a single unified
 * `state` object and an updater that merges updates into the existing state, as
 * opposed to replacing it.
 *
 * ```js
 * const [state, setState] = useMergeState({ name: 'Betsy', age: 24 })
 *
 * setState({ name: 'Johan' }) // { name: 'Johan', age: 24 }
 *
 * setState(state => ({ age: state.age + 10 })) // { name: 'Johan', age: 34 }
 * ```
 *
 * @param initialState The initial state object
 */
export default function useMergeState<TState extends {}>(initialState: TState): [TState, MergeStateSetter<TState>];
export {};
