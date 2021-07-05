## Animation Bus 🚌

[![npm version](https://badge.fury.io/js/animation-bus.svg)](https://badge.fury.io/js/animation-bus)

Define different animation stops relative to another value.

## Install

`yarn add animation-bus`

`npm install animation-bus --save`

```html
<script src="https://unpkg.com/animation-bus/dist/animation-bus.js"></script>
(UMD library exposed as `AnimationBus`)
```

### Example

```js
import AnimationBus from 'animation-bus'

const scrollElements = document.querySelectorAll('[data-scroll-bus]')
const windowFactor = 0.5
const elementFactor = 0.5
let isTicking = false
let scrollTop, scrollBottom

// Define animation stops
const animations = [{
  prop: 'backgroundColor',
  stops: [
    [-300, '#b4da55'],
    [0, '#2ea8ff'],
    [300, '#b4da55']
  ]
}, {
  prop: 'scale',
  stops: [
    [-300, 0.25],
    [0, 1],
    [300, 0.25]
  ]
}, {
  prop: 'opacity',
  stops: [
    [-300, 0],
    [0, 1],
    [300, 0]
  ]
}]

// Define animation stops
const origin = (element) => {
  const windowOffset = window.innerHeight * windowFactor
  const elementOffset = element.offsetHeight * elementFactor
  return scrollTop + windowOffset - elementOffset - element.offsetTop
}

// Instantiate a new animation bus
const animationBus = new AnimationBus({ animations, origin })

// Listen for window scroll and apply transforms to elements
function scrollHandler() {
  for (let i = 0; i < scrollElements.length; i++) {
    animationBus.applyStyles(scrollElements[i])
  }
  isTicking = false
}

window.addEventListener('scroll', function () {
  scrollTop = pageYOffset
  if (!isTicking) {
    window.requestAnimationFrame(scrollHandler)
  }
  isTicking = true
}, false);
```

## Thank You

Huge thank you to [Darin Reid](https://github.com/elcontraption) and all of his work on [Flickity Transformer](https://github.com/elcontraption/flickity-transformer), as well as [this amazing gist](https://github.com/gpiffault) by [Grégoire Piffault](https://github.com/gpiffault). Most of the code in here is heavily inspired by what these developers have previously done.
