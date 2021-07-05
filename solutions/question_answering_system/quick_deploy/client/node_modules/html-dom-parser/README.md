# html-dom-parser

[![NPM](https://nodei.co/npm/html-dom-parser.png)](https://nodei.co/npm/html-dom-parser/)

[![NPM version](https://img.shields.io/npm/v/html-dom-parser.svg)](https://www.npmjs.com/package/html-dom-parser)
[![Build Status](https://travis-ci.org/remarkablemark/html-dom-parser.svg?branch=master)](https://travis-ci.org/remarkablemark/html-dom-parser)
[![Coverage Status](https://coveralls.io/repos/github/remarkablemark/html-dom-parser/badge.svg?branch=master)](https://coveralls.io/github/remarkablemark/html-dom-parser?branch=master)
[![Dependency status](https://david-dm.org/remarkablemark/html-dom-parser.svg)](https://david-dm.org/remarkablemark/html-dom-parser)
[![NPM downloads](https://img.shields.io/npm/dm/html-dom-parser.svg?style=flat-square)](https://www.npmjs.com/package/html-dom-parser)

HTML to DOM parser that works on both the server (Node.js) and the client (browser):

```
HTMLDOMParser(string[, options])
```

It converts an HTML string to a JavaScript object that describes the DOM tree.

#### Example:

```js
var parse = require('html-dom-parser');
parse('<div>text</div>');
```

Output:

```
[ { type: 'tag',
    name: 'div',
    attribs: {},
    children:
     [ { data: 'text',
         type: 'text',
         next: null,
         prev: null,
         parent: [Circular] } ],
    next: null,
    prev: null,
    parent: null } ]
```

[Repl.it](https://repl.it/@remarkablemark/html-dom-parser) | [JSFiddle](https://jsfiddle.net/remarkablemark/ff9yg1yz/) | [Examples](https://github.com/remarkablemark/html-dom-parser/tree/master/examples)

## Installation

[NPM](https://www.npmjs.com/package/html-dom-parser):

```sh
$ npm install html-dom-parser --save
```

[Yarn](https://yarnpkg.com/package/html-dom-parser):

```sh
$ yarn add html-dom-parser
```

[CDN](https://unpkg.com/html-dom-parser/):

```html
<script src="https://unpkg.com/html-dom-parser@latest/dist/html-dom-parser.js"></script>
<script>
  window.HTMLDOMParser(/* string */);
</script>
```

## Usage

Import the module:

```js
// CommonJS
var parse = require('html-dom-parser');

// ES Modules
import parse from 'html-dom-parser';
```

Parse markup:

```js
parse('<p class="primary" style="color: skyblue;">Hello world</p>');
```

Output:

```
[ { type: 'tag',
    name: 'p',
    attribs: { class: 'primary', style: 'color: skyblue;' },
    children:
     [ { data: 'Hello world',
         type: 'text',
         next: null,
         prev: null,
         parent: [Circular] } ],
    next: null,
    prev: null,
    parent: null } ]
```

The _server parser_ is a wrapper of [htmlparser2](https://github.com/fb55/htmlparser2)'s `parseDOM`; the _client parser_ mimics the server parser by using the [DOM](https://developer.mozilla.org/docs/Web/API/Document_Object_Model/Introduction) API.

## Testing

Run server and client tests:

```sh
$ npm test
```

Run server tests with coverage:

```sh
$ npm run test:server:coverage

# generate html report
$ npm run test:server:coverage:report
```

Run client tests:

```sh
$ npm run test:client
```

Lint files:

```sh
$ npm run lint

# fix lint errors
$ npm run lint:fix
```

Test TypeScript declaration file for style and correctness:

```sh
$ npm run dtslint
```

## Release

Only collaborators with credentials can release and publish:

```sh
$ npm run release
$ git push --follow-tags && npm publish
```

## Special Thanks

- [Contributors](https://github.com/remarkablemark/html-dom-parser/graphs/contributors)
- [htmlparser2](https://github.com/fb55/htmlparser2)

## License

[MIT](https://github.com/remarkablemark/html-dom-parser/blob/master/LICENSE)
