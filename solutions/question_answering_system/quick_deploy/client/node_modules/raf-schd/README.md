# raf-schd

A scheduler based on [`requestAnimationFrame`](https://developer.mozilla.org/en-US/docs/Web/API/window/requestAnimationFrame). It throttles calls to a function and only invokes it with the latest argument in the frame period.


[![Build Status](https://travis-ci.org/alexreardon/raf-schd.svg?branch=master)](https://travis-ci.org/alexreardon/raf-schd) [![dependencies](https://david-dm.org/alexreardon/raf-schd.svg)](https://david-dm.org/alexreardon/raf-schd) [![npm](https://img.shields.io/npm/v/raf-schd.svg)](https://www.npmjs.com/package/raf-schd) [![SemVer](https://img.shields.io/badge/SemVer-2.0.0-brightgreen.svg)](http://semver.org/spec/v2.0.0.html)


```js
import rafSchedule from 'raf-schd';

const expensiveFn = (arg) => {
  //...
  console.log(arg);
}

const schedule = rafSchedule(expensiveFn);

schedule('foo');
schedule('bar');
schedule('baz');

// animation frame fires

// => 'baz'
```

## Why?

`raf-schd` supports the use case where you only want to perform an action in an animation frame with the latest value. This an **extremely** useful performance optmisation.

### Without `raf-schd`

> Optimised scroll listener example taken from [MDN](https://developer.mozilla.org/en-US/docs/Web/Events/scroll)

```js
var last_known_scroll_position = 0;
var ticking = false;

function doSomething(scroll_pos) {
  // do something with the scroll position
}

window.addEventListener('scroll', function(e) {
  last_known_scroll_position = window.scrollY;
  if (!ticking) {
    window.requestAnimationFrame(function() {
      doSomething(last_known_scroll_position);
      ticking = false;
    });
  }
  ticking = true;
});
```

### With `raf-schd`

```js
import rafSchedule from 'raf-schd';

function doSomething(scroll_pos) {
  // do something with the scroll position
}

const schedule = rafSchedule(doSomething);

window.addEventListener('scroll', function() {
  schedule(window.scrollY);
});
```

## Types

### `rafSchduler`

```js
type rafSchedule = (fn: Function) => ResultFn

// Adding a .cancel property to the WrapperFn

type WrapperFn = (...arg: any[]) => number;
type CancelFn = {|
  cancel: () => void,
|};
type ResultFn = WrapperFn & CancelFn;
```

At the top level `raf-schd` accepts any function a returns a new `ResultFn` (a function that wraps your original function). When executed, the `ResultFn` returns a `number`. This number is the animation frame id. You can cancel a frame using the `.cancel()` property on the `ResultFn`.

The `ResultFn` will execute your function with the **latest arguments** provided to it on the next animation frame.

### Throttled with latest argument

```js
import rafSchedule from 'raf-schd';

const doSomething = () => {...};

const schedule = rafSchedule(doSomething);

schedule(1, 2);
schedule(3, 4);
schedule(5, 6);

// animation frame fires

// do something called with => '5, 6'
```

### Cancelling a frame

#### `.cancel`

`raf-schd` adds a `.cancel` property to the `ResultFn` so that it can be easily cancelled. The frame will only be cancelled if it has not yet executed.

```js
const scheduled = rafSchedule(doSomething);

schedule('foo');

scheduled.cancel();

// now doSomething will not be executed in the next animation frame
```

#### `cancelAnimationFrame`

You can use [`cancelAnimationFrame`](https://developer.mozilla.org/en-US/docs/Web/API/Window/cancelAnimationFrame) directly to cancel a frame if you like. You can do this because you have the `frameId`.

```js
const scheduled = rafSchedule(doSomething);

const frameId = schedule('foo');

cancelAnimationFrame(frameId);

// now doSomething will not be executed in the next animation frame
```

## Is this a `throttle`, `debounce` or something else?

`raf-schd` is closer to `throttle` than it is `debounce`. It is not like `debounce` because it does not wait for a period of quiet before firing the function.

Lets take a look at the characteristics of this library:

### Similarities to `throttle`

- It batches multiple calls into a single event
- It only executes the wrapped function with the latest argument
- It will not execute anything if the function is not invoked
- One invokation of a scheduled function always results in at least one function call, unless canceled. This is `throttle` with tail calls enabled.

### Differences to `throttle`

- Rather than throttling based on time (such as `200ms`, this library throttles based on `requestAnimationFrame`. This allows the browser to control how many frames to provide per second to optimise rendering.
- Individual frames of `raf-schd` can be canceled using `cancelAnimationFrame` as it returns the frame id.

## Testing your code

If you want to really ensure that your code is working how you intend it to - use [`raf-stub`](https://github.com/alexreardon/raf-stub) to test your animation frame logic.

## Installation

```bash
# yarn
yarn add raf-schd

# npm
npm install raf-schd --save
```

## Module usage

### ES6 module

```js
import rafSchedule from 'raf-schd';
```

### CommonJS

If you are in a CommonJS environment (eg [Node](https://nodejs.org)), then **you will need add `.default` to your import**:

```js
const rafSchedule = require('raf-schd').default;
```
