# API Reference
The components and higher-order components (HOCs) described below are publicly exposed in the top-level module. Other components should be considered private and subject to change without notice.

#### [Components](#components)
- [`<Typeahead>`](#typeahead)
- [`<AsyncTypeahead>`](#asynctypeahead)
- [`<Highlighter>`](#highlighter)
- [`<Input>`](#input)
- [`<Menu>`](#menu)
- [`<MenuItem>`](#menuitem)
- [`<TypeaheadInputSingle>` & `<TypeaheadInputMulti>`](#typeaheadinputsingle--typeaheadinputmulti)
- [`<TypeaheadMenu>`](#typeaheadmenu)
- [`<Token>`](#token)

#### [Higher-Order Components](#higher-order-components)
- [`asyncContainer`](#asynccontainer)
- [`hintContainer`](#hintcontainer)
- [`menuItemContainer`](#menuitemcontainer)
- [`tokenContainer`](#tokencontainer)

## Components
A subset of props are documented below, primarily those expecting functions. See the [props documentation](Props.md) for the full list of options.

### `<Typeahead>`
The primary component provided by the module.

#### Props
Name | Type | Default | Description
-----|------|---------|------------
`allowNew` | `boolean\|function` | `false` | Specifies whether or not arbitrary, user-defined options may be added to the result set. New entries will be included when the trimmed input is truthy and there is no exact match in the result set.<br><br>If a function is specified, allows for a callback to decide whether the new entry menu item should be included in the results list. The callback should return a boolean value:<br><br><pre>`(results: Array<Object\|string>, props: Object) => boolean`</pre>
`filterBy` | `Array<string>\|function` | | See full documentation in the [Filtering section](Filtering.md#filterby-arraystring--function).
`labelKey` | `string\|function` | | See full documentation in the [Rendering section](Rendering.md#labelkey-string--function).
`renderInput` | `function` | | See full documentation in the [Rendering section](Rendering.md#renderinputinputprops-object-state-object).
`renderMenu` | `function` | | See full documentation in the [Rendering section](Rendering.md#rendermenuresults-arrayobjectstring-menuprops-object-state-object).
`renderMenuItemChildren` | `function` | | See full documentation in the [Rendering section](Rendering.md#rendermenuitemchildrenoption-objectstring-props-object-index-number).
`renderToken` | `function` | | See full documentation in the [Rendering section](Rendering.md#rendertokenoption-objectstring-props-object-index-number).
`onChange` | `function` | | Invoked when the set of selections changes (ie: an item is added or removed). For consistency, `selected` is always an array of selections, even if multi-selection is not enabled. <br><br><pre>`(selected: Array<Object\|string>) => void`</pre>
`onInputChange` | `function` | | Invoked when the input value changes. Receives the string value of the input (`text`), as well as the original event. <br><br><pre>`(text: string, event: Event) => void`</pre>
`onBlur`, `onFocus`, `onKeyDown` | `function` | | As with a normal text input, these are called when the typeahead input has blur, focus, or keydown events. <br><br><pre>`(event: Event) => void`</pre>
`onMenuToggle` | `function` | | Invoked when menu visibility changes. <br><br><pre>`(isOpen: boolean) => void`</pre>
`onPaginate` | `function` | | Invoked when the pagination menu item is clicked. Receives an event as the first argument and the number of shown results as the second. <br><br><pre>`(event: Event, shownResults: number) => void`</pre>

### `<AsyncTypeahead>`
An enhanced version of the normal `Typeahead` component for use when performing asynchronous searches. Provides debouncing of user input, optional query caching, and search prompt, empty results, and pending request behaviors.

```jsx
<AsyncTypeahead
  isLoading={this.state.isLoading}
  labelKey={option => `${option.login}`}
  onSearch={(query) => {
    this.setState({isLoading: true});
    fetch(`https://api.github.com/search/users?q=${query}`)
      .then(resp => resp.json())
      .then(json => this.setState({
        isLoading: false,
        options: json.items,
      }));
  }}
  options={this.state.options}
  labelKey={option => `${option.login}`}
/>
```

Note that this component is the same as:
```jsx
import {asyncContainer, Typeahead} from 'react-bootstrap-typeahead';

const AsyncTypeahead = asyncContainer(Typeahead);
```

#### Props
Name | Type | Default | Description
-----|------|---------|------------
`isLoading` (required) | `boolean` | `false` | Whether or not an asynchronous request is in progress.
`onSearch` (required) | `function` | | Callback to perform when the search is executed, where `query` is the input string.<br><br><pre>`(query: string) => void`</pre>

### `<Highlighter>`
Component for highlighting substring matches in the menu items.

```jsx
<Typeahead
  ...
  renderMenuItemChildren={(option, props, idx) => (
    <Highlighter search={props.text}>
      {option[props.labelKey]}
    </Highlighter>
  )}
/>
```

#### Props
Name | Type | Default | Description
-----|------|---------|------------
`search` (required) | `string` | | The substring to look for. This value should correspond to the input text of the typeahead and can be obtained via the `onInputChange` prop or from the `text` property of props being passed down via `renderMenu` or `renderMenuItemChildren`.
`highlightClassName` | `string` | `'rbt-highlight-text'` | Classname applied to the highlighted text.

### `<Input>`
Abstract `<input>` component that handles an `inputRef` prop and is used as the basis for both single- and multi-select input components.

### `<Menu>`
Provides the markup for a Bootstrap menu, along with some extra functionality for specifying a label when there are no results.

Name | Type | Default | Description
-----|------|---------|------------
`emptyLabel` | `string\|element` | `'No matches found.'` | Message to display in the menu if there are no valid results.
`maxHeight` | `string` | `'300px'` | Maximum height of the dropdown menu.


### `<MenuItem>`
Provides the markup for a Bootstrap menu item, but is wrapped with the `menuItemContainer` HOC to ensure proper behavior within the typeahead context. Provided for use if a more customized `Menu` is desired.

#### Props
Name | Type | Default | Description
-----|------|---------|------------
`option` (required) | `Object\|string` | | The data item to be displayed.
`position` | `number` | | The position of the item as rendered in the menu. Allows the top-level `Typeahead` component to be be aware of the item's position despite any custom ordering or grouping in `renderMenu`. **Note:** The value must be a unique, zero-based, sequential integer for proper behavior when keying through the menu.

### `<TypeaheadInputSingle>` & `<TypeaheadInputMulti>`
Input components that handles single- and multi-selections, respectively. In the multi-select component, selections are rendered as children.

#### Props
Name | Type | Default | Description
-----|------|---------|------------
`disabled` | `boolean` | `false` | Whether or not the input component is disabled.

### `<TypeaheadMenu>`
The default menu which is rendered by the `Typeahead` component. Can be used in a custom `renderMenu` function for wrapping or modifying the props passed to it without having to re-implement the default functionality.

#### Props
Name | Type | Default | Description
-----|------|---------|------------
`newSelectionPrefix` | `string` | | Provides the ability to specify a prefix before the user-entered text to indicate that the selection will be new. No-op unless `allowNew={true}`.
`paginationText` | `string` | | Prompt displayed when large data sets are paginated.
`renderMenuItemChildren` | `function` | | Provides a hook for customized rendering of menu item contents.

### `<Token>`
Individual token component, most commonly for use within `renderToken` to customize the `Token` contents.

#### Props
Name | Type | Default | Description
-----|------|---------|------------
`option` (required) | `Object\|string` | | The data item to be displayed.
`disabled` | `boolean` | `false` | Whether the token is in a disabled state. If `true` it will not be interactive or removeable.
`href` | `string` | | If provided, the token will be rendered with an `<a>` tag and `href` attribute.
`readOnly` | `boolean` | `false` | Whether the token is in a read-only state. If `true` it will not be removeable, but it will be interactive if provided an `href`.
`tabIndex` | `number` | `0` | Allows the tabindex to be set if something other than the default is desired.

## Higher-Order Components

### `asyncContainer`
The HOC used in [`AsyncTypeahead`](#asynctypeahead).

### `hintContainer`
Provides hinting functionality when using a custom input (via `renderInput`).

```jsx
import { FormControl } from 'react-bootstrap';
import { hintContainer, Typeahead } from 'react-bootstrap-typeahead';

const HintedFormControl = hintContainer(FormControl);

<Typeahead
  renderInput={(inputProps) => (
    <HintedFormControl {...inputProps} />
  )}
/>
```


### `menuItemContainer`
Connects individual menu items with the main typeahead component via context and abstracts a lot of complex functionality required for behaviors like keying through the menu and input hinting. Also provides `onClick` behavior and active state.

If you use your own menu item components (in `renderMenu` for example), you are strongly advised to wrap them with this HOC:

```jsx
import { MenuItem } from 'react-bootstrap';
import { Menu, menuItemContainer, Typeahead } from 'react-bootstrap-typeahead';

const TypeaheadMenuItem = menuItemContainer(MenuItem);

<Typeahead
  renderMenu={(results, menuProps) => (
    <Menu {...menuProps}>
      {results.map((result, props) => (
        <TypeaheadMenuItem>
          {result.label}
        </TypeaheadMenuItem>
      ))}
    </Menu>
  )}
/>
```

### `tokenContainer`
Encapsulates keystroke and outside click behaviors used in `Token`. Useful if you want completely custom markup for the token.

```jsx
const MyCustomToken = tokenContainer(props => (
  // Your token code...
));

<Typeahead
  multiple
  options={options}
  renderToken={(option, props, index) => (
    <MyCustomToken onRemove={props.onRemove} option={option} />
  )}
/>
```
