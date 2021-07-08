export declare class ObservableMap<K, V> extends Map<K, V> {
    private readonly listener;
    constructor(listener: (map: ObservableMap<K, V>) => void, init?: Iterable<Readonly<[K, V]>>);
    set(key: K, value: V): this;
    delete(key: K): boolean;
    clear(): void;
}
/**
 * Create and return a [Map](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map) that triggers rerenders when it's updated.
 *
 * ```tsx
 * const customerAges = useMap<number>([
 *  ['john', 24],
 *  ['betsy', 25]
 * ]);
 *
 * return (
 *  <>
 *    {Array.from(ids, ([name, age]) => (
 *      <div>
 *        {name}: {age}. <button onClick={() => ids.delete(name)}>X</button>
 *      </div>
 *    )}
 *  </>
 * )
 * ```
 *
 * @param init initial Map entries
 */
declare function useMap<K, V>(init?: Iterable<Readonly<[K, V]>>): ObservableMap<K, V>;
export default useMap;
