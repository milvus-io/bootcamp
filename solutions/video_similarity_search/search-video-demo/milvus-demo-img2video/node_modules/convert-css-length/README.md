# convert-css-length
Convert between css lengths e.g. em->px or px->rem

Conversions between em, ex, rem, px are supported. PRs welcome if
you need support for more esoteric length units.

*[Note: algorithm was originally ported from Compass] (https://github.com/Compass/compass/blob/master/core/stylesheets/compass/typography/_units.scss)*

## Install
`npm install convert-css-length`

## Usage
```javascript
import convertLength from 'convert-css-length';

// Set the baseFontSize for your project. Defaults to 16px (also the
// browser default).
var convert = convertLength('21px');

// Convert rem to px.
convert('1rem', 'px');
// ---> 21px

// Convert px to em.
convert('30px', 'em');
// ---> 1.42857em

// Convert em to pixels using fromContext.
// em(s) are relative to the font-size at the same element. If you're setting an em on a element whose font-size
// is different than the base font size, you'll need to pass that font-size as the third parameter.
// Or just use rem instead which sizes everything relative to the base node.
convert('1em', 'px', '14px')
// ---> 14px
```
