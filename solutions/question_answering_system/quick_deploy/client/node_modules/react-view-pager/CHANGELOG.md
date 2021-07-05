## CHANGELOG

### 0.6.0

Update dependencies

Add peer dependencies for React 16

### 0.5.1

Update react-motion [#63](https://github.com/souporserious/react-view-pager/pull/63)

### 0.5.0

Support React 15.5.0 PropTypes [#58](https://github.com/souporserious/react-view-pager/pull/58)

Update react-motion and resize-observer-polyfill dependency version

### 0.5.0-prerelease.10

Fix build on NPM

### 0.5.0-prerelease.9

Replace `minivents` with `mitt`

Add `tabbable` so we can trap focus within current slides

### 0.5.0-prerelease.8

Use `ResizeObserver.default` if available. Fixes [#48](https://github.com/souporserious/react-view-pager/issues/48)

### 0.5.0-prerelease.7

Only require `ResizeObserver` polyfill when `window` is available

Stop observing view after it has been removed

### 0.5.0-prerelease.6

Apply spring config prop in `Frame` component

### 0.5.0-prerelease.5

Fix old `destroy` method left over from window resize

Listen for hydration event to update `Frame` component size

### 0.5.0-prerelease.4

Use [resize-observer-polyfill](https://github.com/que-etc/resize-observer-polyfill) to make sure the pager is always calculated with proper view dimensions

Hide view visually until it has been added to the pager

General cleanup

### 0.5.0-prerelease.3

emit `viewChange` when `viewsToShow` options have changed

`beforeViewChange` -> `onViewChange`

`afterViewChange` -> `onRest`

Allow `autoSize` prop to size `width` and/or `height`

Use `verticalAlign: top` on views to keep them in frame when using `autoSize`

Added mandatory `View` component that replaces previous views that were cloned. This will help us keep some control over props that need to be applied.

### 0.5.0-prerelease.2

Fixes updating props after mount

Fixes the indices in callbacks to return proper indices in view

Replaced `from`, `to` object returned in `beforeViewChange` callback with `currentIndicies`

No more absolute positioning! This is cool because now we can use things like flexbox and not get weird values due to absolute positioned views.

### 0.5.0-prerelease.1

General cleanup

Fixes Avoid call to window in window-less environment [#28](https://github.com/souporserious/react-view-pager/pull/28)

Fixes `beforeViewChange` being called twice

Fixes `onScroll` callback to allow use of `setState`

Moved main props to the `Track` component

Added `ViewPager` wrapper component. It was a hard decision to add another component, but this will allow some cool animations and other features.

Added `AnimatedView` component to allow animations that are relative to the pager.

Updated `animation-bus` to `0.2.0`

### 0.5.0-prerelease

Major Update again, sorry for the big changes. This has been a rough road, but I feel it has finally smoothed out and I'm very happy with where everything is at.

**Name Change**
`react-motion-slider` -> `react-view-pager`

**Breaking Changes**

_Props changed_

`currentIndex` && `currentKey` -> `currentView`

`slidesToShow` -> `viewsToShow`

`slidesToMove` -> `viewsToMove`

`vertical` -> `axis`

`autoHeight` -> `autoSize`

_Props added_

`contain`

`animations`

`accessibility`

`onSwipeStart`

`onSwipeMove`

`onSwipeEnd`

`onScroll`

`beforeViewChange`

`afterViewChange`

### 0.4.2

Use constructor in `Slider` to fix undefined props in IE10

### 0.4.1

Fix npm build

### 0.4.0

Refactored all code... again. Props are mostly the same, some new ones added. Future changes will be documented better.

### 0.3.0

**Breaking Changes**
Upgraded to React Motion 0.3.0

### 0.2.0

**Breaking Changes**
Refactored almost all code, somewhat similiar API, touch support and some other props were removed for now, will be back in soon

Slider now moves directly to new slides rather than running through everything in between

If using a custom component be sure to pass the `style` prop passed in to apply proper styles for moving slides around. Look in example folder for implementation.

### 0.1.0

**Breaking Changes**
`defaultSlide` prop is now `currentKey`, pass the slides key to navigate to it

exposed as the component `Slider` now instead of and an object
