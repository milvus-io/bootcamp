# html-react-parser

[![NPM](https://nodei.co/npm/html-react-parser.png)](https://nodei.co/npm/html-react-parser/)

[![NPM version](https://img.shields.io/npm/v/html-react-parser.svg)](https://www.npmjs.com/package/html-react-parser)
[![Build Status](https://travis-ci.org/remarkablemark/html-react-parser.svg?branch=master)](https://travis-ci.org/remarkablemark/html-react-parser)
[![Coverage Status](https://coveralls.io/repos/github/remarkablemark/html-react-parser/badge.svg?branch=master)](https://coveralls.io/github/remarkablemark/html-react-parser?branch=master)
[![Dependency status](https://david-dm.org/remarkablemark/html-react-parser.svg)](https://david-dm.org/remarkablemark/html-react-parser)
[![NPM downloads](https://img.shields.io/npm/dm/html-react-parser.svg?style=flat-square)](https://www.npmjs.com/package/html-react-parser)
[![Financial Contributors on Open Collective](https://opencollective.com/html-react-parser/all/badge.svg?label=financial+contributors)](https://opencollective.com/html-react-parser)

HTML to React parser that works on both the server (Node.js) and the client (browser):

```
HTMLReactParser(string[, options])
```

It converts an HTML string to one or more [React elements](https://reactjs.org/docs/react-api.html#creating-react-elements). There's also an option to [replace an element](#replacedomnode) with your own.

#### Example:

```js
var parse = require('html-react-parser');
parse('<div>text</div>'); // equivalent to `React.createElement('div', {}, 'text')`
```

[CodeSandbox](https://codesandbox.io/s/940pov1l4w) | [JSFiddle](https://jsfiddle.net/remarkablemark/7v86d800/) | [Repl.it](https://repl.it/@remarkablemark/html-react-parser) | [Examples](https://github.com/remarkablemark/html-react-parser/tree/master/examples)

## Installation

[NPM](https://www.npmjs.com/package/html-react-parser):

```sh
$ npm install html-react-parser --save
```

[Yarn](https://yarnpkg.com/package/html-react-parser):

```sh
$ yarn add html-react-parser
```

[CDN](https://unpkg.com/html-react-parser/):

```html
<!-- HTMLReactParser depends on React -->
<script src="https://unpkg.com/react@16/umd/react.production.min.js"></script>
<script src="https://unpkg.com/html-react-parser@latest/dist/html-react-parser.min.js"></script>
<script>
  window.HTMLReactParser(/* string */);
</script>
```

## Usage

Import the module:

```js
// ES Modules
import parse from 'html-react-parser';

// CommonJS
const parse = require('html-react-parser');
```

Parse single element:

```js
parse('<h1>single</h1>');
```

Parse multiple elements:

```js
parse('<li>Item 1</li><li>Item 2</li>');
```

Since adjacent elements are parsed as an array, make sure to render them under a parent node:

```jsx
<ul>
  {parse(`
    <li>Item 1</li>
    <li>Item 2</li>
  `)}
</ul>
```

Parse nested elements:

```js
parse('<body><p>Lorem ipsum</p></body>');
```

Parse element with attributes:

```js
parse(
  '<hr id="foo" class="bar" data-attr="baz" custom="qux" style="top:42px;">'
);
```

### Options

#### replace(domNode)

The `replace` callback allows you to swap an element with another React element.

The first argument is an object with the same output as [htmlparser2](https://github.com/fb55/htmlparser2)'s [domhandler](https://github.com/fb55/domhandler#example):

```js
parse('<br>', {
  replace: function(domNode) {
    console.dir(domNode, { depth: null });
  }
});
```

Console output:

```js
{ type: 'tag',
  name: 'br',
  attribs: {},
  children: [],
  next: null,
  prev: null,
  parent: null }
```

The element is replaced only if a _valid_ React element is returned:

```js
parse('<p id="replace">text</p>', {
  replace: domNode => {
    if (domNode.attribs && domNode.attribs.id === 'replace') {
      return React.createElement('span', {}, 'replaced');
    }
  }
});
```

Here's an [example](https://repl.it/@remarkablemark/html-react-parser-replace-example) that modifies an element but keeps the children:

```jsx
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import parse, { domToReact } from 'html-react-parser';

const html = `
  <p id="main">
    <span class="prettify">
      keep me and make me pretty!
    </span>
  </p>
`;

const options = {
  replace: ({ attribs, children }) => {
    if (!attribs) return;

    if (attribs.id === 'main') {
      return <h1 style={{ fontSize: 42 }}>{domToReact(children, options)}</h1>;
    }

    if (attribs.class === 'prettify') {
      return (
        <span style={{ color: 'hotpink' }}>
          {domToReact(children, options)}
        </span>
      );
    }
  }
};

console.log(renderToStaticMarkup(parse(html, options)));
```

Output:

```html
<h1 style="font-size:42px">
  <span style="color:hotpink">
    keep me and make me pretty!
  </span>
</h1>
```

Here's an [example](https://repl.it/@remarkablemark/html-react-parser-56) that excludes an element:

```jsx
parse('<p><br id="remove"></p>', {
  replace: ({ attribs }) => attribs && attribs.id === 'remove' && <Fragment />
});
```

### library

The `library` option allows you to specify which component library is used to create elements. React is used by default if this option is not specified.

Here's an example showing how to use Preact:

```js
parse('<br>', {
  library: require('preact')
});
```

Or, using a custom library:

```js
parse('<br>', {
  library: {
    cloneElement: () => {
      /* ... */
    },
    createElement: () => {
      /* ... */
    },
    isValidElement: () => {
      /* ... */
    }
  }
});
```

## FAQ

#### Is this library XSS safe?

No, this library is _**not**_ [XSS (cross-site scripting)](https://wikipedia.org/wiki/Cross-site_scripting) safe. See [#94](https://github.com/remarkablemark/html-react-parser/issues/94).

#### Does this library sanitize invalid HTML?

No, this library does _**not**_ perform HTML sanitization. See [#124](https://github.com/remarkablemark/html-react-parser/issues/124), [#125](https://github.com/remarkablemark/html-react-parser/issues/125), and [#141](https://github.com/remarkablemark/html-react-parser/issues/141).

#### Are `<script>` tags parsed?

Although `<script>` tags and their contents are rendered on the server-side, they're not evaluated on the client-side. See [#98](https://github.com/remarkablemark/html-react-parser/issues/98).

#### Why aren't my HTML attributes getting called?

This is because [inline event handlers](https://developer.mozilla.org/docs/Web/Guide/Events/Event_handlers) like `onclick` are parsed as a _string_ instead of a _function_. See [#73](https://github.com/remarkablemark/html-react-parser/issues/73).

#### The parser throws an error.

Check if your arguments are valid. Also, see ["Does this library sanitize invalid HTML?"](https://github.com/remarkablemark/html-react-parser#does-this-library-sanitize-invalid-html).

#### Does this library support SSR?

Yes, this library supports server-side rendering on Node.js. See [demo](https://repl.it/@remarkablemark/html-react-parser-SSR).

## Benchmarks

```sh
$ npm run test:benchmark
```

Here's an example output of the benchmarks run on a MacBook Pro 2017:

```
html-to-react - Single x 415,186 ops/sec ±0.92% (85 runs sampled)
html-to-react - Multiple x 139,780 ops/sec ±2.32% (87 runs sampled)
html-to-react - Complex x 8,118 ops/sec ±2.99% (82 runs sampled)
```

## Contributors

### Code Contributors

This project exists thanks to all the people who contribute. [[Contribute](https://github.com/remarkablemark/html-react-parser/blob/master/CONTRIBUTING.md)].

[![Code Contributors](https://opencollective.com/html-react-parser/contributors.svg?width=890&button=false)](https://github.com/remarkablemark/html-react-parser/graphs/contributors)

### Financial Contributors

Become a financial contributor and help us sustain our community. [[Contribute](https://opencollective.com/html-react-parser/contribute)]

#### Individuals

[![Financial Contributors - Individuals](https://opencollective.com/html-react-parser/individuals.svg?width=890)](https://opencollective.com/html-react-parser)

#### Organizations

Support this project with your organization. Your logo will show up here with a link to your website. [[Contribute](https://opencollective.com/html-react-parser/contribute)]

[![Financial Contributors - Organization 0](https://opencollective.com/html-react-parser/organization/0/avatar.svg)](https://opencollective.com/html-react-parser/organization/0/website)
[![Financial Contributors - Organization 1](https://opencollective.com/html-react-parser/organization/1/avatar.svg)](https://opencollective.com/html-react-parser/organization/1/website)
[![Financial Contributors - Organization 2](https://opencollective.com/html-react-parser/organization/2/avatar.svg)](https://opencollective.com/html-react-parser/organization/2/website)
[![Financial Contributors - Organization 3](https://opencollective.com/html-react-parser/organization/3/avatar.svg)](https://opencollective.com/html-react-parser/organization/3/website)
[![Financial Contributors - Organization 4](https://opencollective.com/html-react-parser/organization/4/avatar.svg)](https://opencollective.com/html-react-parser/organization/4/website)
[![Financial Contributors - Organization 5](https://opencollective.com/html-react-parser/organization/5/avatar.svg)](https://opencollective.com/html-react-parser/organization/5/website)
[![Financial Contributors - Organization 6](https://opencollective.com/html-react-parser/organization/6/avatar.svg)](https://opencollective.com/html-react-parser/organization/6/website)
[![Financial Contributors - Organization 7](https://opencollective.com/html-react-parser/organization/7/avatar.svg)](https://opencollective.com/html-react-parser/organization/7/website)
[![Financial Contributors - Organization 8](https://opencollective.com/html-react-parser/organization/8/avatar.svg)](https://opencollective.com/html-react-parser/organization/8/website)
[![Financial Contributors - Organization 9](https://opencollective.com/html-react-parser/organization/9/avatar.svg)](https://opencollective.com/html-react-parser/organization/9/website)

## Support

- [Patreon](https://b.remarkabl.org/patreon)
- [Open Collective](https://b.remarkabl.org/open-collective-html-react-parser)
- [Ko-fi](https://b.remarkabl.org/ko-fi)
- [Tidelift](https://b.remarkabl.org/tidelift-html-react-parser)
- [Liberapay](https://b.remarkabl.org/liberapay)
- [Teepsring](https://b.remarkabl.org/teespring)

## License

[MIT](https://github.com/remarkablemark/html-react-parser/blob/master/LICENSE)
