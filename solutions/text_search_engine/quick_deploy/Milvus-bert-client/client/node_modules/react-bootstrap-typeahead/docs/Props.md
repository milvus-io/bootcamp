# Props

### `<Typeahead>`
Name | Type | Default | Description
-----|------|---------|------------
align | One of: `justify`, `left`, `right` | 'justify' | Specify menu alignment. The default value is `justify`, which makes the menu as wide as the input and truncates long values. Specifying `left` or `right` will align the menu to that side and the width will be determined by the length of menu item values.
allowNew | boolean or function | false | Allows the creation of new selections on the fly. Any new items will be added to the list of selections, but not the list of original options unless handled as such by `Typeahead`'s parent. The newly added item will *always* be returned as an object even if the other options are simply strings, so be sure your `onChange` callback can handle this. If a function is specified, it will be used to determine whether a custom option should be included. The return value should be `true` or `false`.
autoFocus | boolean | false | Autofocus the input when the component initially mounts.
bsSize | one of: `'large'`, `'lg'`, `'small'`, `'sm'` | | Specify the size of the input.
caseSensitive | boolean | false | Whether or not filtering should be case-sensitive.
clearButton | boolean | false | Displays a button to clear the input when there are selections.
defaultInputValue | string | '' | The initial value displayed in the text input.
defaultOpen | boolean | false | Whether or not the menu is displayed upon initial render.
defaultSelected | array | `[]` | Specify any pre-selected options. Use only if you want the component to be uncontrolled.
disabled | boolean | | Whether to disable the input. Will also disable selections when `multiple={true}`.
dropup | boolean | false | Specify whether the menu should appear above the input.
emptyLabel | node | 'No matches found.' | Message displayed in the menu when there are no valid results.
filterBy | function or array | `[]` | Either an array of fields in `option` to search, or a custom filtering callback. See the [Filtering](https://github.com/ericgio/react-bootstrap-typeahead/blob/master/docs/Filtering.md#filterby) page for more info.
flip | boolean | false | Whether or not to automatically adjust the position of the menu when it reaches the viewport boundaries.
highlightOnlyResult | boolean | false | Highlights the menu item if there is only one result and allows selecting that item by hitting enter. Does not work with `allowNew`.
id `required` | string or number | | An html id attribute, required for assistive technologies such as screen readers.
ignoreDiacritics | boolean | true | Whether the filter should ignore accents and other diacritical marks.
inputProps | object | {} | Props to be applied directly to the input. `onBlur`, `onChange`, `onFocus`, and `onKeyDown` are ignored.
isInvalid | boolean | false | Adds the `is-invalid` classname to the `form-control`. Only affects Bootstrap 4.
isLoading | boolean | false | Indicate whether an asynchronous data fetch is happening.
isValid | boolean | false | Adds the `is-valid` classname to the `form-control`. Only affects Bootstrap 4.
labelKey | string or function | 'label' | Specify which option key to use for display or a render function. By default, the selector will use the `label` key.
maxHeight | string | '300px' | Maximum height of the dropdown menu.
maxResults | number | 100 | Maximum number of results to display by default. Mostly done for performance reasons so as not to render too many DOM nodes in the case of large data sets.
minLength | number | 0 | Number of input characters that must be entered before showing results.
multiple | boolean | false | Whether or not multiple selections are allowed.
newSelectionPrefix | string | 'New selection:' | Provides the ability to specify a prefix before the user-entered text to indicate that the selection will be new. No-op unless `allowNew={true}`.
onBlur | function | | Invoked when the input is blurred. Receives an event.
onChange | function | | Invoked whenever items are added or removed. Receives an array of the selected options.
onFocus | function | | Invoked when the input is focused. Receives an event.
onInputChange | function | | Invoked when the input value changes. Receives the string value of the input.
onKeyDown | function | | Invoked when a key is pressed. Receives an event.
onMenuToggle | function | | Invoked when menu visibility changes.
onPaginate | function | | Invoked when the pagination menu item is clicked.
open | boolean | | Whether or not the menu should be displayed. `undefined` allows the component to control visibility, while `true` and `false` show and hide the menu, respectively.
options `required` | array | | Full set of options, including any pre-selected options.
paginate | boolean | true | Give user the ability to display additional results if the number of results exceeds `maxResults`.
paginationText | string | 'Display additional results...' | Prompt displayed when large data sets are paginated.
placeholder | string | | Placeholder text for the input.
positionFixed | boolean | false | Whether to use fixed positioning for the menu, which is useful when rendering inside a container with `overflow: hidden;`. Uses absolute positioning by default.
renderInput | function | | Callback for custom input rendering.
renderMenu | function | | Callback for custom menu rendering.
renderMenuItemChildren | function | | Provides a hook for customized rendering of menu item contents.
renderToken | function | | Provides a hook for customized rendering of tokens when multiple selections are enabled.
selected | array | `[]` | The selected option(s) displayed in the input. Use this prop if you want to control the component via its parent.
selectHintOnEnter | boolean | false | Allows selecting the hinted result by pressing enter.

### `<AsyncTypeahead>`
Name | Type | Default | Description
-----|------|---------|------------
delay | number | 200 | Delay, in milliseconds, before performing search.
isLoading `required` | boolean | | Whether or not a request is currently pending. Necessary for the component to know when new results are available.
onSearch `required` | function | | Callback to perform when the search is executed.
options | array | `[]` | Options to be passed to the typeahead. Will typically be the query results, but can also be initial default options.
promptText | node | 'Type to search...' | Message displayed in the menu when there is no user input.
searchText | node | 'Searching...' | Message to display in the menu while the request is pending.
useCache | bool | true | Whether or not the component should cache query results.

[Next: API Reference](API.md)
