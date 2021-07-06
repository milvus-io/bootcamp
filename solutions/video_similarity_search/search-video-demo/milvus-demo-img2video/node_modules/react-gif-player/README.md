# react-gif-player

![react-gif-player in action](demo.gif)

### [see a live demo here](https://benwiley4000.github.io/react-gif-player/)

Similar to Facebook's GIF toggle UI, this React component displays a still image preview by default, and swaps in an animated GIF when clicked. The images are preloaded as soon as the component mounts, or whenever a new source is passed.

*Note:* Unlike Facebook's UI, which uses an HTML video element to preserve playback progress, this component uses the actual GIF and will be reset on each click.

[![NPM](https://nodei.co/npm/react-gif-player.png)](https://npmjs.org/package/react-gif-player)

## install

```
npm install react-gif-player react react-dom
```

If you're unable to use npm and need production-ready scripts, check out [the releases](https://github.com/benwiley4000/react-gif-player/releases).

## usage

### quick start

```html
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <!-- gifplayer.css v0.4.2 -->
    <link rel="stylesheet" href="https://unpkg.com/react-gif-player@0.4.2/dist/gifplayer.css">
  </head>
  <body>
    <div id="cat"></div>
    <!-- react/react-dom served over CDN -->
    <script src="https://unpkg.com/react@16.3.0-alpha.1/umd/react.development.js"></script>
    <script src="https://unpkg.com/react@16.3.0-alpha.1/umd/react-dom.development.js"></script>
    <!-- gifplayer.js v0.4.2 -->
    <script src="https://unpkg.com/react-gif-player@0.4.2/dist/gifplayer.js"></script>
    <script>
      ReactDOM.render(
        React.createElement(GifPlayer, {
          gif: '/img/cat.gif',
          still: '/img/cat.jpg'
        }),
        document.getElementById('cat')
      );
    </script>
  </body>
</html>

```

### with a module bundler

```javascript
var React = require('react');
var ReactDOM = require('react-dom');
var GifPlayer = require('react-gif-player');

// with JSX
ReactDOM.render(
  <GifPlayer gif="/img/cat.gif" still="/img/cat.jpg" />,
  document.getElementById('cat')
);
```

### options

Options can be passed to the `GifPlayer` element as props.

* `gif`: a *string* address to an animated GIF image.

* `still`: a *string* address to a still preview of the GIF (e.g. JPG, PNG, etc.)

* `autoplay`: a *boolean* which can be set `true` if you want to immediately bombard your user with a moving GIF as soon as it's available

* `onTogglePlay`: a *function* which is called whenever the GIF toggles between playing and paused. Receives one argument, `playing`, which is a *boolean*.
 
* `pauseRef`: a *function* callback is called with another function, `pause` - this can be saved and called later to remotely pause the playing of the GIF, in such cases where that might be desired. For example, you might want to stop the GIF when it scrolls offscreen. The word "ref" is used because its usage pattern is similar to [React element refs](https://reactjs.org/docs/refs-and-the-dom.html):
  ```jsx
  // here's an example
  class MyGifWrapper extends React.Component {
    componentDidMount () {
      addEventListenerWhenGifFlowsOffscreen(this.pauseGif);
    }
 
    render () {
      return (
        <GifPlayer
          gif={src}
          still={still}
          pauseRef={pause => this.pauseGif = pause}
        />
      );
    }
  }
  ```
  Still confused? Here's a [jsfiddle](http://jsfiddle.net/1snhzgo8/7/) which shows how to use `pauseRef`.
 
* Any other attribute available on the HTML `img` tag can be passed as well (excluding `src`, which would be overwritten), though keep in mind React's version of that attribute [may be different than you expect](https://facebook.github.io/react/docs/dom-elements.html#all-supported-html-attributes).

`GifPlayer` expects one or both of the `gif` and `still` props. If one is left out, the other will be used as a fallback.

However, if only a `gif` prop is provided, the first frame will be extracted and used as the still preview as soon as the GIF image has fully loaded.

## generating still frame at build time

The disadvantage of not providing a `still` prop, even though a stand-in will be generated, is that your GIF must fully load before the still frame appears instead of the (likely slowly moving) GIF.

One streamlined way to generate a still frame ahead of time is to incorporate the [gif-frames module](https://github.com/benwiley4000/gif-frames), which has only pure JavaScript dependencies, into your build process.

e.g.

```javascript
var gifFrames = require('gif-frames');
var fs = require('fs');

gifFrames({ url: 'src/image.gif', frames: 0 }).then(function (frameData) {
  frameData[0].getImageStream().pipe(fs.createWriteStream('build/still.jpg'));
});
```

If you need finer-tuned control over image quality, you can try [Gifsicle](https://www.lcdf.org/gifsicle/).

## styles

**Important:** In order for the default styles to be used, **dist/gifplayer.css** must be included in your HTML.

CSS styles can be overridden easily. To add a border around the image, try including this CSS **after** including the default styles:

```css
.gif_player img {
  border: 3px solid cornflowerblue;
}
```

## usage with sass

If you preprocess your styles with Sass, you can have more powerful control via Sass variables. The defaults are located at the top of [**src/GifPlayer.scss**](src/GifPlayer.scss):

```scss
$gif_btn_bg_base_color: #000 !default;
$gif_btn_bg_opacity: 0.5 !default;
$gif_btn_bg_opacity_hover: 0.7 !default;
// ...etc
```
The `!default` flag means that declaring alternatives **before** including the default styles will override them.

```scss
// Include var overrides before default styles import
$gif_btn_bg_base_color: gold;
$gif_btn_text_color: cornflowerblue;
$gif_btn_font_family: serif;

// Using webpack css/sass module import syntax
@import '~react-gif-player/src/GifPlayer';

// include other overrides afterward
.gif_player {
  margin: 1rem;

  img {
    border: 2px solid #222;
  }
}
```

## development

For building and testing instructions, see [CONTRIBUTING.md](CONTRIBUTING.md).
