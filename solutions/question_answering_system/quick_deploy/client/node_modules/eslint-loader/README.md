<div align="center">
  <a href="https://github.com/eslint/eslint"><img width="200" height="200" src="https://cdn.worldvectorlogo.com/logos/eslint.svg"></a>
  <a href="https://github.com/webpack/webpack"><img width="200" height="200" src="https://webpack.js.org/assets/icon-square-big.svg"></a>
</div>

[![npm][npm]][npm-url]
[![node][node]][node-url]
[![deps][deps]][deps-url]
[![tests][tests]][tests-url]
[![coverage][cover]][cover-url]
[![chat][chat]][chat-url]
[![size][size]][size-url]

# eslint-loader

> A ESlint loader for webpack

## Install

```bash
npm install eslint-loader --save-dev
```

**Note**: You also need to install `eslint` from npm, if you haven't already:

```bash
npm install eslint --save-dev
```

## Usage

In your webpack configuration:

```js
module.exports = {
  // ...
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        loader: 'eslint-loader',
        options: {
          // eslint options (if necessary)
        },
      },
    ],
  },
  // ...
};
```

When using with transpiling loaders (like `babel-loader`), make sure they are in correct order (bottom to top). Otherwise files will be checked after being processed by `babel-loader`:

```js
module.exports = {
  // ...
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: ['babel-loader', 'eslint-loader'],
      },
    ],
  },
  // ...
};
```

To be safe, you can use `enforce: 'pre'` section to check source files, not modified by other loaders (like `babel-loader`):

```js
module.exports = {
  // ...
  module: {
    rules: [
      {
        enforce: 'pre',
        test: /\.js$/,
        exclude: /node_modules/,
        loader: 'eslint-loader',
      },
      {
        test: /\.js$/,
        exclude: /node_modules/,
        loader: 'babel-loader',
      },
    ],
  },
  // ...
};
```

## Options

You can pass [eslint options](http://eslint.org/docs/developer-guide/nodejs-api#cliengine) using standard webpack [loader options](https://webpack.js.org/configuration/module/#useentry).

Note that the config option you provide will be passed to the `CLIEngine`. This is a different set of options than what you'd specify in `package.json` or `.eslintrc`. See the [eslint docs](http://eslint.org/docs/developer-guide/nodejs-api#cliengine) for more detail.

### `cache`

- Type: `Boolean|String`
- Default: `false`

This option will enable caching of the linting results into a file. This is particularly useful in reducing linting time when doing a full build.

This can either be a `boolean` value or the cache directory path(ex: `'./.eslint-loader-cache'`).

If `cache: true` is used, the cache is written to the `./node_modules/.cache/eslint-loader` directory. This is the recommended usage.

```js
module.exports = {
  entry: '...',
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        loader: 'eslint-loader',
        options: {
          cache: true,
        },
      },
    ],
  },
};
```

### `eslintPath`

- Type: `String`
- Default: `eslint`

Path to `eslint` instance that will be used for linting. If the `eslintPath` is a folder like a official eslint, or specify a `formatter` option. Now you dont have to install `eslint`.

```js
module.exports = {
  entry: '...',
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        loader: 'eslint-loader',
        options: {
          eslintPath: path.join(__dirname, 'reusable-eslint'),
        },
      },
    ],
  },
};
```

### `fix`

- Type: `Boolean`
- Default: `false`

This option will enable [ESLint autofix feature](http://eslint.org/docs/user-guide/command-line-interface#fix).

**Be careful: this option will change source files.**

```js
module.exports = {
  entry: '...',
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        loader: 'eslint-loader',
        options: {
          fix: true,
        },
      },
    ],
  },
};
```

### `formatter`

- Type: `String|Function`
- Default: `stylish`

This option accepts a function that will have one argument: an array of eslint messages (object). The function must return the output as a string. You can use official [eslint formatters](https://eslint.org/docs/user-guide/formatters/).

```js
module.exports = {
  entry: '...',
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        loader: 'eslint-loader',
        options: {
          // several examples !

          // default value
          formatter: 'stylish',

          // community formatter
          formatter: require('eslint-friendly-formatter'),

          // custom formatter
          formatter: function(results) {
            // `results` format is available here
            // http://eslint.org/docs/developer-guide/nodejs-api.html#executeonfiles()

            // you should return a string
            // DO NOT USE console.*() directly !
            return 'OUTPUT';
          },
        },
      },
    ],
  },
};
```

### Errors and Warning

**By default the loader will auto adjust error reporting depending on eslint errors/warnings counts.** You can still force this behavior by using `emitError` **or** `emitWarning` options:

#### `emitError`

- Type: `Boolean`
- Default: `false`

Will always return errors, if this option is set to `true`.

```js
module.exports = {
  entry: '...',
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        loader: 'eslint-loader',
        options: {
          emitError: true,
        },
      },
    ],
  },
};
```

#### `emitWarning`

- Type: `Boolean`
- Default: `false`

Will always return warnings, if option is set to `true`. **If you're using hot module replacement, you may wish to enable this in development, or else updates will be skipped when there's an eslint error.**

```js
module.exports = {
  entry: '...',
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        loader: 'eslint-loader',
        options: {
          emitWarning: true,
        },
      },
    ],
  },
};
```

#### `failOnError`

- Type: `Boolean`
- Default: `false`

Will cause the module build to fail if there are any errors, if option is set to `true`.

```js
module.exports = {
  entry: '...',
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        loader: 'eslint-loader',
        options: {
          failOnError: true,
        },
      },
    ],
  },
};
```

#### `failOnWarning`

- Type: `Boolean`
- Default: `false`

Will cause the module build to fail if there are any warnings, if option is set to `true`.

```js
module.exports = {
  entry: '...',
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        loader: 'eslint-loader',
        options: {
          failOnWarning: true,
        },
      },
    ],
  },
};
```

#### `quiet`

- Type: `Boolean`
- Default: `false`

Will process and report errors only and ignore warnings, if this option is set to `true`.

```js
module.exports = {
  entry: '...',
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        loader: 'eslint-loader',
        options: {
          quiet: true,
        },
      },
    ],
  },
};
```

#### `outputReport`

- Type: `Boolean|Object`
- Default: `false`

Write the output of the errors to a file, for example a checkstyle xml file for use for reporting on Jenkins CI.

The `filePath` is an absolute path or relative to the webpack config: `output.path`. You can pass in a different `formatter` for the output file, if none is passed in the default/configured formatter will be used.

```js
module.exports = {
  entry: '...',
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        loader: 'eslint-loader',
        options: {
          outputReport: {
            filePath: 'checkstyle.xml',
            formatter: 'checkstyle',
          },
        },
      },
    ],
  },
};
```

## Gotchas

### NoEmitOnErrorsPlugin

`NoEmitOnErrorsPlugin` is now automatically enabled in webpack 4, when mode is either unset, or set to production. So even ESLint warnings will fail the build. No matter what error settings are used for `eslint-loader`, except if `emitWarning` enabled.

### Defining `configFile` or using `eslint -c path/.eslintrc`

Bear in mind that when you define `configFile`, `eslint` doesn't automatically look for `.eslintrc` files in the directory of the file to be linted. More information is available in official eslint documentation in section [_Using Configuration Files_](http://eslint.org/docs/user-guide/configuring#using-configuration-files). See [#129](https://github.com/webpack-contrib/eslint-loader/issues/129).

## Changelog

[Changelog](CHANGELOG.md)

## License

[MIT](./LICENSE)

[npm]: https://img.shields.io/npm/v/eslint-loader.svg
[npm-url]: https://npmjs.com/package/eslint-loader
[node]: https://img.shields.io/node/v/eslint-loader.svg
[node-url]: https://nodejs.org
[deps]: https://david-dm.org/webpack-contrib/eslint-loader.svg
[deps-url]: https://david-dm.org/webpack-contrib/eslint-loader
[tests]: https://dev.azure.com/webpack-contrib/eslint-loader/_apis/build/status/webpack-contrib.eslint-loader?branchName=master
[tests-url]: https://dev.azure.com/webpack-contrib/eslint-loader/_build/latest?definitionId=4&branchName=master
[cover]: https://codecov.io/gh/webpack-contrib/eslint-loader/branch/master/graph/badge.svg
[cover-url]: https://codecov.io/gh/webpack-contrib/eslint-loader
[chat]: https://badges.gitter.im/webpack/webpack.svg
[chat-url]: https://gitter.im/webpack/webpack
[size]: https://packagephobia.now.sh/badge?p=eslint-loader
[size-url]: https://packagephobia.now.sh/result?p=eslint-loader
