import { DependencyList, EffectCallback } from 'react';
export declare type EffectHook = (effect: EffectCallback, deps?: DependencyList) => void;
export declare type IsEqual<TDeps extends DependencyList> = (nextDeps: TDeps, prevDeps: TDeps) => boolean;
export declare type CustomEffectOptions<TDeps extends DependencyList> = {
    isEqual: IsEqual<TDeps>;
    effectHook?: EffectHook;
};
/**
 * a useEffect() hook with customized depedency comparision
 *
 * @param effect The effect callback
 * @param dependencies A list of dependencies
 * @param isEqual A function comparing the next and previous dependencyLists
 */
declare function useCustomEffect<TDeps extends DependencyList = DependencyList>(effect: EffectCallback, dependencies: TDeps, isEqual: IsEqual<TDeps>): void;
/**
 * a useEffect() hook with customized depedency comparision
 *
 * @param effect The effect callback
 * @param dependencies A list of dependencies
 * @param options
 * @param options.isEqual A function comparing the next and previous dependencyLists
 * @param options.effectHook the underlying effect hook used, defaults to useEffect
 */
declare function useCustomEffect<TDeps extends DependencyList = DependencyList>(effect: EffectCallback, dependencies: TDeps, options: CustomEffectOptions<TDeps>): void;
export default useCustomEffect;
