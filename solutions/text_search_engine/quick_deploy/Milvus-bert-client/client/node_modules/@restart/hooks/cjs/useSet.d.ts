export declare class ObservableSet<V> extends Set<V> {
    private readonly listener;
    constructor(listener: (map: ObservableSet<V>) => void, init?: Iterable<V>);
    add(value: V): this;
    delete(value: V): boolean;
    clear(): void;
}
/**
 * Create and return a [Set](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Set) that triggers rerenders when it's updated.
 *
 * ```ts
 * const ids = useSet<number>([1,2,3,4]);
 *
 * return (
 *  <>
 *    {Array.from(ids, id => (
 *      <div>
 *        id: {id}. <button onClick={() => ids.delete(id)}>X</button>
 *      </div>
 *    )}
 *  </>
 * )
 * ```
 *
 * @param init initial Set values
 */
declare function useSet<V>(init?: Iterable<V>): ObservableSet<V>;
export default useSet;
