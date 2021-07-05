/// <reference types="react" />
/**
 * Exactly the same as `useRef` except that the initial value is set via a
 * factroy function. Useful when the default is relatively costly to construct.
 *
 *  ```ts
 *  const ref = useRefWithInitialValueFactory<ExpensiveValue>(() => constructExpensiveValue())
 *
 *  ```
 *
 * @param initialValueFactory A factory function returning the ref's default value
 * @category refs
 */
export default function useRefWithInitialValueFactory<T>(initialValueFactory: () => T): import("react").MutableRefObject<T>;
