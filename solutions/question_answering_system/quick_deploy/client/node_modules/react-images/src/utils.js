// @flow

// ==============================
// NO OP
// ==============================

export const noop = () => {};

// ==============================
// Class Name Prefixer
// ==============================

type State = { [key: string]: boolean };
type List = Array<string>;

export const CLASS_PREFIX = 'react-images';

/**
 String representation of component state for styling with class names.

 Expects an array of strings OR a string/object pair:
 - className(['comp', 'comp-arg', 'comp-arg-2'])
   @returns 'react-images__comp react-images__comp-arg react-images__comp-arg-2'
 - className('comp', { some: true, state: false })
   @returns 'react-images__comp react-images__comp--some'
*/
export function className(name: string | List, state?: State): string {
  const arr: List = Array.isArray(name) ? name : [name];

  // loop through state object, remove falsey values and combine with name
  if (state && typeof name === 'string') {
    for (let key in state) {
      if (state.hasOwnProperty(key) && state[key]) {
        arr.push(`${name}--${key}`);
      }
    }
  }

  // prefix everything and return a string
  return arr.map(cn => `${CLASS_PREFIX}__${cn}`).join(' ');
}

// ==============================
// Touch Capability Detector
// ==============================

export function isTouch() {
  try {
    document.createEvent('TouchEvent');
    return true;
  } catch (e) {
    return false;
  }
}
