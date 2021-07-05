# react-property

[![NPM](https://nodei.co/npm/react-property.png)](https://nodei.co/npm/react-property/)

[![NPM version](https://img.shields.io/npm/v/react-property.svg)](https://www.npmjs.com/package/react-property)

HTML and SVG DOM property configs used by React.

## Install

```sh
# with npm
$ npm install react-property --save

# with yarn
$ yarn add react-property
```

## Usage

Import main module:

```js
// CommonJS
const reactProperty = require('react-property');

// ES Modules
import reactProperty from 'react-property';
```

Main module exports:

```js
{
  html: {
    autofocus: {
      attributeName: 'autofocus',
      propertyName: 'autoFocus',
      mustUseProperty: false,
      hasBooleanValue: true,
      hasNumericValue: false,
      hasPositiveNumericValue: false,
      hasOverloadedBooleanValue: false
    },
    // ...
  },
  svg: {
    // ...
  },
  properties: {
    // ...
  },
  isCustomAttribute: [Function: bound test]
}
```

You may also import what you need:

```js
const HTMLDOMPropertyConfig = require('react-property/lib/HTMLDOMPropertyConfig');
const injection = require('react-property/lib/injection');
```

## Layout

```
.
├── index.js
└── lib
    ├── HTMLDOMPropertyConfig.js
    ├── SVGDOMPropertyConfig.js
    └── injection.js
```

## License

MIT. See [license](https://github.com/facebook/react/blob/15-stable/LICENSE) from original project.
