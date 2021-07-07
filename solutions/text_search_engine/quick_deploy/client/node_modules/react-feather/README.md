## React Feather Icons

[![npm version](https://img.shields.io/npm/v/react-feather.svg?style=flat-square)](https://www.npmjs.com/package/react-feather)
[![npm downloads](https://img.shields.io/npm/dm/react-feather.svg?style=flat-square)](https://www.npmjs.com/package/react-feather)

#### What is react-feather?
react-feather is a collection of simply beautiful open source icons for React.js. Each icon is designed on a 24x24 grid with an emphasis on simplicity, consistency and readability.

[Explore icons](https://bit.dev/feathericons/react-feather) and play with examples.
<p align="center">
  <a href="https://bit.dev/feathericons/react-feather"><img src="https://i.imagesup.co/images2/371e9a2b398a5d7cef54a956d959e4534b290a55.gif"></a>
</p>

#### Based on Feather Icons  ```v4.22.1```

#### Version `2.0.0` adds support for tree-shaking, to help you reduce bundle size.

### Installation
    npm install react-feather --save

### Usage

```javascript
import React from 'react';
import { Camera } from 'react-feather';

const App = () => {
  return <Camera />
};

export default App;
```

Icons can be configured with inline props:
```javascript
<Camera color="red" size={48} />
```

If you can't use ES6 imports, it's possible to include icons from the compiled folder ./dist.
```javascript
var Camera = require('react-feather/dist/icons/camera').default;

var MyComponent = React.createClass({
  render: function () {
    return (
      <Camera />
    );
  }
});
```
You can also include the whole icon pack:

```javascript
import React from 'react';
import * as Icon from 'react-feather';

const App = () => {
  return <Icon.Camera />
};

export default App;
```
