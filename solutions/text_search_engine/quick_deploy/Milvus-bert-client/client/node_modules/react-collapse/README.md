# react-collapse [![npm](https://img.shields.io/npm/v/react-collapse.svg?style=flat-square)](https://www.npmjs.com/package/react-collapse)

[![Gitter](https://img.shields.io/gitter/room/nkbt/help.svg?style=flat-square)](https://gitter.im/nkbt/help)

[![CircleCI](https://img.shields.io/circleci/project/nkbt/react-collapse.svg?style=flat-square&label=nix-build)](https://circleci.com/gh/nkbt/react-collapse)
[![Dependencies](https://img.shields.io/david/nkbt/react-collapse.svg?style=flat-square)](https://david-dm.org/nkbt/react-collapse)
[![Dev Dependencies](https://img.shields.io/david/dev/nkbt/react-collapse.svg?style=flat-square)](https://david-dm.org/nkbt/react-collapse#info=devDependencies)

Component-wrapper for collapse animation for elements with variable (and dynamic) height

![React Collapse](example/react-collapse.gif)

## Demo

[http://nkbt.github.io/react-collapse](http://nkbt.github.io/react-collapse)

## Codepen demo

[http://codepen.io/nkbt/pen/MarzEg](http://codepen.io/nkbt/pen/MarzEg?editors=101)

## Installation

### NPM

```sh
npm install --save react-collapse
```

### yarn

```sh
yarn add react-collapse 
```

### 1998 Script Tag:
```html
<script src="https://unpkg.com/react/umd/react.production.min.js"></script>
<script src="https://unpkg.com/react-collapse/build/react-collapse.min.js"></script>
(Module exposed as `ReactCollapse`)
```


## Usage

Default behaviour, never unmounts content

```js
import {Collapse} from 'react-collapse';

// ...
<Collapse isOpened={true || false}>
  <div>Random content</div>
</Collapse>
```

If you want to unmount collapsed content, use `Unmount` component provided as:

```js
import {UnmountClosed} from 'react-collapse';

// ...
<UnmountClosed isOpened={true || false}>
  <div>Random content</div>
</UnmountClosed>
```

Example [example/App/AutoUnmount.js](example/App/AutoUnmount.js)


## Options


### `isOpened`: PropTypes.boolean.isRequired

Expands or collapses content.


### `children`: PropTypes.node.isRequired

One or multiple children with static, variable or dynamic height.

```js
<Collapse isOpened={true}>
  <p>Paragraph of text</p>
  <p>Another paragraph is also OK</p>
  <p>Images and any other content are ok too</p>
  <img src="nyancat.gif" />
</Collapse>
```

### `theme`: PropTypes.objectOf(PropTypes.string)

It is possible to set `className` for extra `div` elements that ReactCollapse creates.

Example:
```js
<Collapse theme={{collapse: 'foo', content: 'bar'}}>
  <div>Customly animated container</div>
</Collapse>
```

Default values:
```js
const theme = {
  collapse: 'ReactCollapse--collapse',
  content: 'ReactCollapse--content'
}
```

Which ends up in the following markup:
```html
<div class="ReactCollapse--collapse">
  <div class="ReactCollapse--content">
    {children}
  </div>
</div>
```

**IMPORTANT**: these are not style objects, but class names!


### `onRest`, `onWork`: PropTypes.func

Callback functions, triggered when animation has completed (`onRest`) or has just started (`onWork`)

Both functions are called with argument:
```js
const arg = {
  isFullyOpened: true || false, // `true` only when Collapse reached final height
  isFullyClosed: true || false, // `true` only when Collapse is fully closed and height is zero
  isOpened: true || false, // `true` if Collapse has any non-zero height
  containerHeight: 123, // current pixel height of Collapse container (changes until reaches `contentHeight`)
  contentHeight: 123 // determined height of supplied Content 
}
```

```js
<Collapse onRest={console.log} onWork={console.log}>
  <div>Container text</div>
</Collapse>
```

Example [example/App/Hooks.js](example/App/Hooks.js)

### `initialStyle`: PropTypes.shape

```js
initialStyle: PropTypes.shape({
  height: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
  overflow: PropTypes.string
})
```

You may control initial element style, for example to force initial animation from 0 to height by using `initialStyle={{height: '0px', overflow: 'hidden'}}`

**IMPORTANT** Any updates to this prop will be ignored after first render.
Default value is determined based on initial `isOpened` value:
```js
  initialStyle = props.isOpened
    ? {height: 'auto', overflow: 'initial'}
    : {height: '0px', overflow: 'hidden'};
```

Example: [example/App/ForceInitialAnimation.js](example/App/ForceInitialAnimation.js)

### `checkTimeout`: PropTypes.number

Number in `ms`. 

Collapse will check height after thins timeout to determine if animation is completed, the shorter the number - the faster `onRest` will be triggered and the quicker `hight: auto` will be applied. The downside - more calculations.
Default value is: `50`. 


### Pass-through props

**IMPORTANT** Collapse does not support any pass-through props, so any non-supported props will be ignored

Because we need to have control over when `Collapse` component is updated and it is not possible or very hard to achieve when any random props can be passed to the component.


## Behaviour notes

- initially opened Collapse elements will be statically rendered with no animation. You can override this behaviour by using `initialStyle` prop

- due to the complexity of margins and their potentially collapsible nature, `ReactCollapse` does not support (vertical) margins on their children. It might lead to the animation "jumping" to its correct height at the end of expanding. To avoid this, use padding instead of margin.
  See [#101](https://github.com/nkbt/react-collapse/issues/101) for more details


## Migrating from `v4` to `v5`

`v5` was another complete rewrite that happened quite a while ago, it was published as `@nkbt/react-collapse` and tested in real projects for a long time and now fully ready to be used.

In the most common scenario upgrade is trivial (add CSS transition to collapse element), but if you were using a few deprecated props - there might be some extra work required.

Luckily almost every deprecated callback or prop has fully working analogue in `v5`. Unfortunately not all of them could maintain full backward compatibility, so please check this migration guide below.

### 1. Change in behaviour

New Collapse does not implement animations anymore, it only determines `Content` height and updates `Collapse` wrapper height to match it.
Only after `Collapse` height reaches `Content` height (animation finished), Collapse's style is updated to have `height: auto; overflow: initial`.

The implications is that you will need to update your CSS with transition:
```css
.ReactCollapse--collapse {
  transition: height 500ms;
}
```
**IMPORTANT**: without adding css transition there will be no animation and component will open/close instantly.

### 2. New props or updated props

- `onRest`/`onWork` callbacks (see above for full description). Though `onRest` did exist previously, now it is called with arguments containing few operational params and flags.

- `initialStyle` you may control initial element style, for example to force initial animation from 0 to height by using `initialStyle={{height: '0px', overflow: 'hidden'}}`
    **IMPORTANT** Any updates to this prop will be ignored after first render.
    Default value is:
    ```js
      initialStyle = props.isOpened
        ? {height: 'auto', overflow: 'initial'}
        : {height: '0px', overflow: 'hidden'};
    ```

- `checkTimeout` number in `ms`. Collapse will check height after thins timeout to determine if animation is completed, the shorter the number - the faster `onRest` will be triggered and the quicker `hight: auto` will be applied. The downside - more calculations.
    Default value is: `50`. 

### 3. Deprecated props (not available in `v5`)
- ~~Pass-through props~~ - any unknown props passed to `Collapse` component will be ignored

- ~~hasNestedCollapse~~ - no longer necessary, as v5 does not care if there are nested Collapse elements, see [example/App/Nested.js](example/App/Nested.js)

- ~~fixedHeight~~ - no longer necessary, just set whatever height you need for content element and Collapse will work as expected, see [example/App/VariableHeight.js](example/App/VariableHeight.js)

- ~~springConfig~~ - as new Collapse relies on simple CSS transitions (or your own implementation of animation) and does not use react-collapse, springConfig is no longer necessary. You can control control animation with css like
    ```css
    .ReactCollapse--collapse {
      transition: height 500ms;
    }
    ```

- ~~forceInitialAnimation~~ - you can use new prop `initialStyle={{height: '0px', overflow: 'hidden'}}` instead, so when new height will be set after component is rendered - it should naturally animate.

- ~~onMeasure~~ - please use `onRest` and `onWork` instead and pick `contentHeight` from argument
    ```js
    <Collapse
      onRest={({contentHeight}) => console.log(contentHeight)}
      onWork={({contentHeight}) => console.log(contentHeight)}>
      <div>content</div>
    </Collapse>
    ```
- ~~onRender~~ - since animations are fully controlled by external app (e.g. with CSS) we no draw each frame and do not actually re-render component anymore, so it is impossible to have `onRender` callback

## Development and testing

Currently is being developed and tested with the latest stable `Node` on `OSX`.

To run example covering all `ReactCollapse` features, use `yarn start`, which will compile `example/Example.js`

```bash
git clone git@github.com:nkbt/react-collapse.git
cd react-collapse
yarn install
yarn start

# then
open http://localhost:8080
```

## Tests

```bash
# to run ESLint check
yarn lint

# to run tests
yarn test
```

## License

MIT
