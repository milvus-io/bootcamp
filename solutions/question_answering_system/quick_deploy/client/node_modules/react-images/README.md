# React Images

A mobile-friendly, highly customizable, carousel component for displaying media in ReactJS.

### Browser support

Should work in every major browser... maybe even IE10 and IE11?

### Getting Started

Start by installing `react-images`

```bash
npm install react-images
```
or
```bash
yarn add react-images
```

**If you were using `0.x` versions:** library was significantly rewritten for `1.x` version and contains several breaking changes.
The best way to upgrade is to read the docs and follow the examples.

Please note that the default footer parses HTML automatically (such as `<b>I'm bold!</b>`) but it **does not implement any form of XSS or sanitisation**. You should do that yourself before passing it into the caption field of react-images.

### Using the Carousel

Import the carousel from `react-images` at the top of a
component and then use it in the render function.

```jsx
import React from 'react';
import Carousel from 'react-images';

const images = [{ source: 'path/to/image-1.jpg' }, { source: 'path/to/image-2.jpg' }];

class Component extends React.Component {
  render() {
    return <Carousel views={images} />;
  }
}
```

### Using the Modal

Import the modal and optionally the modal gateway from
`react-images` at the top of a component and then use it in
the render function.

The `ModalGateway` will insert the modal just before the
end of your `<body />` tag.

```jsx
import React from 'react';
import Carousel, { Modal, ModalGateway } from 'react-images';

const images = [{ source: 'path/to/image-1.jpg' }, { source: 'path/to/image-2.jpg' }];

class Component extends React.Component {
  state = { modalIsOpen: false };
  toggleModal = () => {
    this.setState(state => ({ modalIsOpen: !state.modalIsOpen }));
  };
  render() {
    const { modalIsOpen } = this.state;

    return (
      <ModalGateway>
        {modalIsOpen ? (
          <Modal onClose={this.toggleModal}>
            <Carousel views={images} />
          </Modal>
        ) : null}
      </ModalGateway>
    );
  }
}
```
