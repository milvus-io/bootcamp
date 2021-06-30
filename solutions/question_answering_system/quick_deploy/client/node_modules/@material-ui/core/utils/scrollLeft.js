"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.detectScrollType = detectScrollType;
exports.getNormalizedScrollLeft = getNormalizedScrollLeft;
// Source from https://github.com/alitaheri/normalize-scroll-left
var cachedType;
/**
 * Based on the jquery plugin https://github.com/othree/jquery.rtl-scroll-type
 *
 * Types of scrollLeft, assiming scrollWidth=100 and direction is rtl.
 *
 * Browser        | Type          | <- Most Left | Most Right -> | Initial
 * -------------- | ------------- | ------------ | ------------- | -------
 * WebKit         | default       | 0            | 100           | 100
 * Firefox/Opera  | negative      | -100         | 0             | 0
 * IE/Edge        | reverse       | 100          | 0             | 0
 */

function detectScrollType() {
  if (cachedType) {
    return cachedType;
  }

  var dummy = document.createElement('div');
  dummy.appendChild(document.createTextNode('ABCD'));
  dummy.dir = 'rtl';
  dummy.style.fontSize = '14px';
  dummy.style.width = '4px';
  dummy.style.height = '1px';
  dummy.style.position = 'absolute';
  dummy.style.top = '-1000px';
  dummy.style.overflow = 'scroll';
  document.body.appendChild(dummy);
  cachedType = 'reverse';

  if (dummy.scrollLeft > 0) {
    cachedType = 'default';
  } else {
    dummy.scrollLeft = 1;

    if (dummy.scrollLeft === 0) {
      cachedType = 'negative';
    }
  }

  document.body.removeChild(dummy);
  return cachedType;
} // Based on https://stackoverflow.com/a/24394376


function getNormalizedScrollLeft(element, direction) {
  var scrollLeft = element.scrollLeft; // Perform the calculations only when direction is rtl to avoid messing up the ltr bahavior

  if (direction !== 'rtl') {
    return scrollLeft;
  }

  var type = detectScrollType();

  switch (type) {
    case 'negative':
      return element.scrollWidth - element.clientWidth + scrollLeft;

    case 'reverse':
      return element.scrollWidth - element.clientWidth - scrollLeft;

    default:
      return scrollLeft;
  }
}