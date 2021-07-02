# Upgrade Guide

- [Version 4.0](Upgrading.md#v40)
- [Version 3.0](Upgrading.md#v30)
- [Version 2.0](Upgrading.md#v20)
- [Version 1.0](Upgrading.md#v10)

## v4.0 Breaking Changes

### Drop support for older versions of React
React and ReactDOM >=16.3 are now required as peer dependencies.

### Props
- `onMenuHide` and `onMenuShow` were removed. Use `onMenuToggle` instead.

### When using custom menu items, you must manually pass a ref to the underlying DOM node.
`tokenContainer` no longer uses `findDOMNode`, and instead passes a `ref` to the wrapped component. If you are using your own menu item component to render the menu, you must forward that ref to the the underlying DOM node.

### `option` is now required for a `Token` to be removeable
Pass `option` as a prop to `Token` (or `tokenContainer` if using your own token markup) so the container can correctly handle the `onRemove` callback.

```jsx
<Typeahead
  ...
  renderToken={(option, props, idx) => (
    <Token
      ...
      onRemove={props.onRemove}
      option={option}>
      {option.label}
    </Token>
  )}
/>
```


### Falsy `emptyLabel` no longer hides the menu when there are no results
This behavior was a legacy workaround introduced before `renderMenu` could return `null`. That is no longer the case and `renderMenu` should now be used to achieve the behavior:

```jsx
<Typeahead
  ...
  renderMenu={(results, menuProps) => {
    if (!results.length) {
      return null;
    }

    return <TypeaheadMenu {...menuProps} />;
  }}
/>
```

### `id` required for assistive technologies
The `menuId` prop has been replaced by `id` and no longer provides a default value. You must provide an id for assistive technologies like screen readers.

### Input `autoComplete` attribute defaults to "off"
Behavior is now correct according to a11y standards, but may result in unexpected behaviors since different browsers handle this attribute differently.

- To keep the previous behavior, pass "nope" value in `inputProps`.

### Updates to the `Overlay` component
Overlay was updated to take advantage of Popper.js' fixed positioning rather than using a portal to attach the menu to the document body. This greatly simplifies the component and gives greater control over styling.

- `bodyContainer` props was removed. Use `positionFixed` instead.
- Use of `.rbt-body-container` for CSS scoping will no longer work. Pass your own scoping classnames to the component instead.

### A11y announcer removed
This piece of functionality is not part of the WAI-ARIA authoring guidelines and was difficult to test and support.

- `a11yNumResults` & `a11yNumSelected` are now no-ops
- If you need this functionality, you can add it yourself as a child (or child function) of the component.

## v3.0 Breaking Changes

### Props
- The `name` prop was deprecated in v2.0 and is now gone.
- Non-string values for the `maxHeight` prop were deprecated in v2.5 and are now no longer allowed.

### Changes to `filterBy` and `renderToken` callback signatures

#### `filterBy`
If you [pass a callback to `filterBy`]((Filtering.md#functionoption-objectstring-text-string)), it will now receive the set of internal props as the second parameter instead of the user-input `text` value:

```jsx
// v2.0
<Typeahead
  ...
  filterBy={(option, text) => {
    // Your own filtering code goes here.
  }}
/>

// v3.0
<Typeahead
  ...
  filterBy={(option, props) => {
    // Your own filtering code goes here.
    // `text` is now `props.text`
  }}
/>
```
#### `renderToken`
Similarly, [`renderToken`](Rendering.md#rendertokenoption-objectstring-onremove-function-index-number) now receives internal props as the second param rather than just the `onRemove` function:

```jsx
// v2.0
<Typeahead
  ...
  multiple
  renderToken={(option, onRemove, index) => {
    // Your own token rendering code.
  }}
/>

// v3.0
<Typeahead
  ...
  multiple
  renderToken={(option, props, index) => {
    // Your own token rendering code.
    // `onRemove` is now `props.onRemove`
  }}
/>
```

### Internal changes & CSS
This version includes some significant internal refactoring in an effort to provide better support for Bootstrap 4. If you have custom CSS that depends on internal (eg: `rbt-*`) classnames, you should check to make sure things still work as you expect.

### Query normalization in `AsyncTypeahead`
`AsyncTypeahead` no longer trims whitespace on or lowercases queries. The original intent was to provide some basic normalization of queries, but this resulted in strange behaviors. You should now do any checking you want, like ignoring queries with only whitespace, in your `onSearch` function.

### Change events no longer triggered by prop changes
The `onChange` and `onInputChange` callbacks were previously called in `componentWillReceiveProps`, which triggered multiple calls and didn't emulate how a normal form element works. These change callbacks are now only triggered by user actions, eg: typing in the input, clicking on a menu item, etc. You may need to update your code if it relied on a change event being triggered due to prop changes.

### Custom menu rendering
Finally, if you use the `renderMenu` prop, a couple changes were made that may affect you:

#### Popper.js for positioning
The typeahead now uses [Popper.js](https://popper.js.org/) (via [`react-popper`](https://github.com/souporserious/react-popper)) for menu positioning. If you're using the provided `Menu` component inside `renderMenu`, simply pass down all the menu props and everything should work fine. If you're using your own component to render the menu, be sure it properly consumes the `innerRef` prop that gets passed down or the component will not work correctly:

```jsx
class MyCustomMenu extends React.Component {
  render() {
    // `innerRef` is passed down by the Popper...
    const {innerRef, ...props} = this.props;

    // ...and must be passed to the `ref` of your custom component.
    return <div {...props} ref={innerRef} />;
  }
}
```

#### Manual handling of pagination option
To make the pagination menu item keyboard-accessible, it is no longer automatically included in the `Menu` component. Instead, it is added to the result set, similar to the custom (`allowNew`) item. That means you must now handle rendering of the pagination item yourself if you want pagination. See [`TypeaheadMenu`](../src/TypeaheadMenu.react.js) for an example of how to do this.

## v2.0
Version 2.0 consists mainly of internal refactoring aimed at reducing parallel code paths and making certain complex feature requests possible. These changes should mostly be transparent, though you may notice that the component behaves a bit differently.

### Breaking Changes
#### `AsyncTypeahead`
The `AsyncTypeahead` component now requires the request state to be managed externally. Use the `isLoading` prop to tell the component if a request is pending or not. See [the example](https://github.com/ericgio/react-bootstrap-typeahead/blob/master/example/examples/AsyncExample.react.js) for an illustration of proper usage.

#### CSS Changes
In an effort to simplify the CSS and as a result of the refactor, class names for the various internal components were changed. This may cause styling to break if you relied on a certain naming scheme. The separate CSS files were also combined into a single file (`Typeahead.css`) to make it easier to include.

### Deprecations
- The `name` prop is now deprecated and will be removed in v3.0. Use `inputProps` to apply props directly to the input instead.

## v1.0
Version 1.0 has a few breaking changes, but upgrading should be relatively painless.

### Importing
The main change affecting all users is that the typeahead is now a property of the module:

```jsx
// v0.10.x
import Typeahead from 'react-bootstrap-typeahead'; // ES2015
var Typeahead = require('react-bootstrap-typeahead').default; // CommonJS

// v1.x
import {Typeahead} from 'react-bootstrap-typeahead'; // ES2015
var Typeahead = require('react-bootstrap-typeahead').Typeahead; // CommonJS
```

### `paginateResults`
This prop was deprecated in v0.9.0 and is now gone.

### `renderMenuItemChildren`
The signature for the `renderMenuItemChildren` callback was changed such that the data item is now the first argument and props are second. This felt more logical and [all such `render` functions](Rendering.md#rendermenu) follow a similar pattern.

```jsx
// v0.10.x
renderMenuItemChildren(props, result, index) {
  // Rendering code here...
}

// v1.x
renderMenuItemChildren(result, props, index) {
  // Rendering code here...
}
```

That should be everything. If you come across something I missed, please open an issue. Thanks!

[Next: Basic Usage](Usage.md)
