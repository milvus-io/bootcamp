# React Photo Gallery

[![Join the chat at https://gitter.im/react-photo-gallery/Lobby](https://badges.gitter.im/react-photo-gallery/Lobby.svg)](https://gitter.im/react-photo-gallery/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![npm version](https://badge.fury.io/js/react-photo-gallery.svg)](https://badge.fury.io/js/react-photo-gallery)
![npm downloads](https://img.shields.io/npm/dt/react-photo-gallery.svg)
[![Build Status](https://travis-ci.org/neptunian/react-photo-gallery.svg?branch=master)](https://travis-ci.org/neptunian/react-photo-gallery)
[![Coverage Status](https://coveralls.io/repos/github/neptunian/react-photo-gallery/badge.svg?branch=master)](https://coveralls.io/github/neptunian/react-photo-gallery?branch=master)
[![Dependency Status](https://david-dm.org/neptunian/react-photo-gallery.svg)](https://david-dm.org/neptunian/react-photo-gallery)

* Responsive, accessible, composable, and customizable image gallery component 
* Maintains the original aspect ratio of your photos
* Creates a masonry or justified grid 
* Supports row or column direction layout
* Provides an image renderer for custom implementation of things like image selection, favorites, captions, etc.
* SSR app compatible

## Preview

| Row       | Column           |
| ------------- |:-------------:|
| <img src="https://live.staticflickr.com/65535/40680327133_6f6218bfa3.jpg" /> | <img src="https://live.staticflickr.com/65535/47594180212_0dc2694eda.jpg"> |


## Installation

```
yarn add react-photo-gallery
```

## API Documentation

[http://neptunian.github.io/react-photo-gallery/](http://neptunian.github.io/react-photo-gallery/)

## CodeSandbox Demos with Example Use Cases

* [Basic Row Layout](https://codesandbox.io/s/9yx911wl9y)
* [Basic Column Layout](https://codesandbox.io/s/r09k1xj614)
* [With Lightbox](https://codesandbox.io/s/5vn3lvz2n4)
* [Dynamic Columns Using columns prop](https://codesandbox.io/s/ll7ym48027)
* [Selection using custom renderImage](https://codesandbox.io/s/o7o241q09)
* [Sortable with drag and drop](https://codesandbox.io/s/8y7n1r9y5j)

To build some examples locally, git clone and run:

```
yarn install
yarn start
```

Then open [`localhost:8000`](http://localhost:8000) in a browser.


## Minimal Setup Example

```jsx

const photos = [
  {
    src: 'http://example.com/example/img1.jpg',
    width: 4,
    height: 3
  },
  {
    src: 'http://example.com/example/img2.jpg',
    width: 1,
    height: 1
  }
];

<Gallery photos={photos} />;

```

## How It Works

### Row Layout
This layout uses an algorithm adapted from the Knuth and Plass line breaking algorithm.  It uses a graph to calculate the single best layout where each photo to break on is represented by a node and each edge is represented by a row. The cost of the edge is determined by the user provided `targetRowHeight` vs the row height calculated if it were to break on this node/photo. What you end up with is a layout with rows that are similar in height and photos that are not being stretched or shrunken abnormally as is what happens in a naive implementation. This solves the issue of panoramas shrinking rows or having stragglers or stretched images at the last row, instead creating a justified grid.  To make sure it's speedy the graph is being built as the shortest path is being calculated so the entire adjacency list is not calculated ahead of time. You can control how many neighboring nodes that Dijkstra's algorithm will search when it's visiting a node by adjusting the `limitNodeSearch` property, but it's recommended you use the default algorithm. See documentation for recommendations.

Inspired by [this blog article](http://blog.vjeux.com/2014/image/google-plus-layout-find-best-breaks.html) and this [Google Photos blog article](https://medium.com/google-design/google-photos-45b714dfbed1) (under 2. Justified Gallery).

### Column Layout
Goes through each column looking for the best place to insert the next photo by finding the shortest column. Not recommended for panorama aspect ratios.

## Thanks

Special thanks to [Christopher Chedeau](https://blog.vjeux.com/) for [writing about this interesting algorithm](http://blog.vjeux.com/2014/image/google-plus-layout-find-best-breaks.html) and whos code served as a starting off point.