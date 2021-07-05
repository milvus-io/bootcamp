import activeElement from './activeElement';
import addClass from './addClass';
import addEventListener from './addEventListener';
import { cancel as cancelAnimationFrame, request as requestAnimationFrame } from './animationFrame';
import closest from './closest';
import contains from './contains';
import style from './css';
import filter from './filterEventHandler';
import hasClass from './hasClass';
import height from './height';
import listen from './listen';
import matches from './matches';
import offset from './offset';
import offsetParent from './offsetParent';
import ownerDocument from './ownerDocument';
import ownerWindow from './ownerWindow';
import position from './position';
import querySelectorAll from './querySelectorAll';
import removeClass from './removeClass';
import removeEventListener from './removeEventListener';
import scrollParent from './scrollParent';
import scrollTop from './scrollTop';
import toggleClass from './toggleClass';
import width from './width';
export { addEventListener, removeEventListener, filter, listen, style, activeElement, ownerDocument, ownerWindow, requestAnimationFrame, cancelAnimationFrame, matches, height, width, offset, offsetParent, position, contains, scrollParent, scrollTop, querySelectorAll, closest, addClass, removeClass, hasClass, toggleClass };
export default {
  addEventListener: addEventListener,
  removeEventListener: removeEventListener,
  filter: filter,
  listen: listen,
  style: style,
  activeElement: activeElement,
  ownerDocument: ownerDocument,
  ownerWindow: ownerWindow,
  requestAnimationFrame: requestAnimationFrame,
  cancelAnimationFrame: cancelAnimationFrame,
  matches: matches,
  height: height,
  width: width,
  offset: offset,
  offsetParent: offsetParent,
  position: position,
  contains: contains,
  scrollParent: scrollParent,
  scrollTop: scrollTop,
  querySelectorAll: querySelectorAll,
  closest: closest,
  addClass: addClass,
  removeClass: removeClass,
  hasClass: hasClass,
  toggleClass: toggleClass
};