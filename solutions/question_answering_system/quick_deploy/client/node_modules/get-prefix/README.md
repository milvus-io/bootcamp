## Get Prefix 1.0.0

Get browser style prefixes for Javascript.

## Install

`npm install get-prefix --save`

`bower install get-prefix --save`

## Example Usage

```javascript
import getPrefix from 'get-prefix'

const div = document.createElement('div')
const transform = getPrefix('transform')

div[transform] = 'rotate(360deg)'
```
