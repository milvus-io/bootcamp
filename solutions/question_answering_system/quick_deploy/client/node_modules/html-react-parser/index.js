var domToReact = require('./lib/dom-to-react');
var htmlToDOM = require('html-dom-parser');

// decode HTML entities by default for `htmlparser2`
var domParserOptions = { decodeEntities: true, lowerCaseAttributeNames: false };

/**
 * Converts HTML string to React elements.
 *
 * @param  {String}   html                    - The HTML string to parse to React.
 * @param  {Object}   [options]               - The parser options.
 * @param  {Function} [options.replace]       - The replace method.
 * @return {JSX.Element|JSX.Element[]|String} - Returns React element(s), string, or empty array.
 */
function HTMLReactParser(html, options) {
  if (typeof html !== 'string') {
    throw new TypeError('First argument must be a string');
  }
  return domToReact(htmlToDOM(html, domParserOptions), options);
}

HTMLReactParser.domToReact = domToReact;
HTMLReactParser.htmlToDOM = htmlToDOM;

// support CommonJS and ES Modules
module.exports = HTMLReactParser;
module.exports.default = HTMLReactParser;
