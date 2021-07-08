# React Bootstrap Typeahead
A [React](https://reactjs.org/)-based typeahead that relies on [Bootstrap](https://getbootstrap.com/) for styling and was originally inspired by Twitter's [typeahead.js](https://github.com/twitter/typeahead.js). It supports both single- and multi-selection and is compliant with [WAI-ARIA authoring practices](https://www.w3.org/TR/wai-aria-practices-1.1/#combobox). Try the [live examples](http://ericgio.github.io/react-bootstrap-typeahead/).

[![npm version](https://img.shields.io/npm/v/react-bootstrap-typeahead.svg?style=flat-square)](https://www.npmjs.com/package/react-bootstrap-typeahead)
[![npm downloads](https://img.shields.io/npm/dm/react-bootstrap-typeahead.svg?style=flat-square)](https://www.npmjs.com/package/react-bootstrap-typeahead)
[![build status](https://img.shields.io/travis/ericgio/react-bootstrap-typeahead/master.svg?style=flat-square)](https://travis-ci.org/ericgio/react-bootstrap-typeahead)
[![Coveralls github](https://img.shields.io/coveralls/github/ericgio/react-bootstrap-typeahead/4.0.svg?style=flat-square)](https://coveralls.io/github/ericgio/react-bootstrap-typeahead)
[![MIT License](https://flat.badgen.net/github/license/ericgio/react-bootstrap-typeahead)](LICENSE.md)

Please note that documentation and examples apply to the most recent release and may no longer be applicable if you're using an outdated version.

## Installation

```
npm install --save react-bootstrap-typeahead
```
or
```
yarn add react-bootstrap-typeahead
```

Include the module in your project:
```jsx
import {Typeahead} from 'react-bootstrap-typeahead'; // ES2015
var Typeahead = require('react-bootstrap-typeahead').Typeahead; // CommonJS
```

#### UMD Build
Development and production builds are included in the NPM package. Alternatively, you can get them from [CDNJS](https://cdnjs.com/libraries/react-bootstrap-typeahead) or [unpkg](https://unpkg.com/react-bootstrap-typeahead/).

## Documentation
- [Basic Usage](docs/Usage.md)
- [Data](docs/Data.md)
- [Filtering](docs/Filtering.md)
- [Rendering](docs/Rendering.md)
- [Public Methods](docs/Methods.md)
- [Props](docs/Props.md)
- [API](docs/API.md)
- [Upgrade Guide](docs/Upgrading.md)

## CSS
While the component relies primarily on Bootstrap for styles, some additional CSS is necessary for everything to behave correctly. You must include the provided CSS file in your project:

```js
// Import as a module in your JS
import 'react-bootstrap-typeahead/css/Typeahead.css';
```
or
```html
<!-- Link as a stylesheet in your HTML -->
<link rel="stylesheet" href="https://unpkg.com/react-bootstrap-typeahead/css/Typeahead.css">
```

### Bootstrap 4
In an effort to fully support Bootstrap 4, this package also contains a CSS file named `Typeahead-bs4.css` that should be included alongside the base CSS file above.

## Examples
Try the [live examples](http://ericgio.github.io/react-bootstrap-typeahead/), which also include code samples. If you'd like to modify the examples, clone the repository and run `npm install` and `npm start` to build the example file. You can then open the HTML file locally in your browser.

## Browser Support
Recent versions of the following browsers are supported:
- Chrome
- Firefox
- IE (>=11)
- Safari

Special thanks to [BrowserStack](https://www.browserstack.com) for providing cross-browser testing support.

[![http://i.imgur.com/9aLP6Fx.png?1](http://i.imgur.com/9aLP6Fx.png?1)](https://www.browserstack.com)

## License
[MIT](https://github.com/ericgio/react-bootstrap-typeahead/blob/master/LICENSE.md)
