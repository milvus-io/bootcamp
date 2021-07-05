// @flow
// parser plugins
// render plugins

import { processStyleName } from './CSSPropertyOperations';

export function fallbacks(node) {
  let hasArray =
    Object.keys(node.style)
      .map(x => Array.isArray(node.style[x]))
      .indexOf(true) >= 0;
  if (hasArray) {
    let { style } = node;
    let flattened = Object.keys(style).reduce((o, key) => {
      o[key] = Array.isArray(style[key])
        ? style[key].join(`; ${processStyleName(key)}: `)
        : style[key];
      return o;
    }, {});
    // todo -
    // flatten arrays which haven't been flattened yet
    return { ...node, style: flattened };
  }
  return node;
}

let contentValues = [
  'normal',
  'none',
  'counter',
  'open-quote',
  'close-quote',
  'no-open-quote',
  'no-close-quote',
  'initial',
  'inherit',
];

export function contentWrap(node) {
  if (node.style.content) {
    let cont = node.style.content;
    if (contentValues.indexOf(cont) >= 0) {
      return node;
    }
    if (/^(attr|calc|counters?|url)\(/.test(cont)) {
      return node;
    }
    if (
      cont.charAt(0) === cont.charAt(cont.length - 1) &&
      (cont.charAt(0) === '"' || cont.charAt(0) === "'")
    ) {
      return node;
    }
    return { ...node, style: { ...node.style, content: '"' + cont + '"' } };
  }
  return node;
}

import prefix from './prefix';

export function prefixes(node) {
  return { ...node, style: prefix({ ...node.style }) };
}
