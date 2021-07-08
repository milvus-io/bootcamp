import { useEffect, useLayoutEffect } from 'react';
var isReactNative = typeof global !== 'undefined' && // @ts-ignore
global.navigator && // @ts-ignore
global.navigator.product === 'ReactNative';
var isDOM = typeof document !== 'undefined';
/**
 * Is `useLayoutEffect` in a DOM or React Native environment, otherwise resolves to useEffect
 * Only useful to avoid the console warning.
 *
 * PREFER `useEffect` UNLESS YOU KNOW WHAT YOU ARE DOING.
 *
 * @category effects
 */

export default isDOM || isReactNative ? useLayoutEffect : useEffect;