# react-fade-in

Dead-simple and opinionated component to fade in an element's children.

![React Fade In](/example/example.gif)

## Installation

`npm install react-fade-in`

## Usage

`react-fade-in`

```
import FadeIn from 'react-fade-in';
// ...
<FadeIn>
  <div>Element 1</div>
  <div>Element 2</div>
  <div>Element 3</div>
  <div>Element 4</div>
  <div>Element 5</div>
  <div>Element 6</div>
</FadeIn>
```

## API

### `FadeIn`

Animates its children, one by one.

> **Note**: To have children animate separately, they must be first-level children of the `<FadeIn>` component (i.e. members of its `props.children`).

#### Props

*   `delay`: Default: 50. Delay between each child's animation in milliseconds.
*   `transitionDuration`: Default: 400. Duration of each child's animation in milliseconds.
*   `className`: No default. Adds a `className` prop to the container div.
*   `childClassName`: No default. Adds a `className` prop to each child div, allowing you to style the direct children of the `FadeIn` component.

Happy fading.
