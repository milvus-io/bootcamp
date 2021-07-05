# React Fullscreen

A React component that sets its children to fullscreen using the Fullscreen API, normalized using [fscreen](https://github.com/rafrex/fscreen).

## Usage

### * Install.
```bash
yarn add react-full-screen
```

### * Import component.
```js
import Fullscreen from "react-full-screen";
```

### * Setup and render.
```jsx
import React, { Component } from "react";
import Fullscreen from "react-full-screen";

class App extends Component {
  constructor(props) {
    super();

    this.state = {
      isFull: false,
    };
  }

  goFull = () => {
    this.setState({ isFull: true });
  }

  render() {
    return (
      <div className="App">
        <button onClick={this.goFull}>
          Go Fullscreen
        </button>

        <Fullscreen
          enabled={this.state.isFull}
          onChange={isFull => this.setState({isFull})}
        >
          <div className="full-screenable-node">
            Hi! This may cover the entire monitor.
          </div>
        </Fullscreen>
      </div>
    );
  }
}

export default App;
```

It is not possible to start in Fullscreen. Fullscreen must be enabled from a user action such as `onClick`.

The reason for keeping track of the current state outside of the component is that the user can choose to leave full screen mode without the action of your application. This is a safety feature of the Fullscreen API.

## Props

### `enabled` *boolean*
Set to `true` when component should go fullscreen.

### `onChange` *function*
Optional callback that gets called when state changes.


## CSS

Class `fullscreen-enabled` will be added to component when it goes fullscreen. If you want to alter child elements when this happens you can use a typical CSS statement.

```css
.my-component {
  background: #fff;
}

.fullscreen-enabled .my-component {
  background: #000;
}
```

## In the wild

Used with [MegamanJS](http://megaman.pomle.com/)
