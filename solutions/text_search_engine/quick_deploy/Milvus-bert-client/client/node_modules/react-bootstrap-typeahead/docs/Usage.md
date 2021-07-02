# Basic Usage
The typeahead behaves similarly to other form elements. It requires an array of data options to be filtered and displayed.
```jsx
<Typeahead
  onChange={(selected) => {
    // Handle selections...
  }}
  options={[ /* Array of objects or strings */ ]}
/>
```

### Single & Multi-Selection
The component provides single-selection by default, but also supports multi-selection. Simply set the `multiple` prop and the component turns into a tokenizer:

```jsx
<Typeahead
  multiple
  onChange={(selected) => {
    // Handle selections...
  }}
  options={[...]}
/>
```

### Controlled vs. Uncontrolled
Similar to other form elements, the typeahead can be [controlled](https://facebook.github.io/react/docs/forms.html#controlled-components) or [uncontrolled](https://facebook.github.io/react/docs/forms.html#uncontrolled-components). Use the `selected` prop to control it via the parent, or `defaultSelected` to optionally set defaults and then allow the component to control itself. Note that the *selections* can be controlled, not the input value.

#### Controlled (Recommended)
```jsx
<Typeahead
  onChange={(selected) => {
    this.setState({selected});
  }}
  options={[...]}
  selected={this.state.selected}
/>
```

#### Uncontrolled
```jsx
<Typeahead
  defaultSelected={[...]}
  onChange={(selected) => {
    // Handle selections...
  }}
  options={[...]}
/>
```

[Next: Data](Data.md)
