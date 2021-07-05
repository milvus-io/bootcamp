var path = require('path');
var webpack = require('webpack');
var TARGET = process.env.TARGET || null;

var config = {
  entry: {
    index: './src/index.js',
  },
  output: {
    path: path.join(__dirname, 'dist'),
    publicPath: 'dist/',
    filename: 'get-prefix.js',
    sourceMapFilename: 'get-prefix.sourcemap.js',
    library: 'getPrefix',
    libraryTarget: 'umd'
  },
  module: {
    loaders: [
      {test: /\.(js|jsx)/, loader: 'babel?stage=0'}
    ]
  },
  plugins: [],
  resolve: {
    extensions: ['', '.js', '.jsx']
  }
};

if(TARGET === 'minify') {
  config.output.filename = 'get-prefix.min.js';
  config.output.sourceMapFilename = 'get-prefix.min.js';
  config.plugins.push(new webpack.optimize.UglifyJsPlugin({
    compress: {
      warnings: false
    },
    mangle: {
      except: ['getPrefix']
    }
  }));
}

module.exports = config;