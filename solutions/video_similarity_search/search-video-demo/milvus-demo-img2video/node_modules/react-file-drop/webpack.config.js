/* eslint-disable import/no-commonjs */
const fs = require('fs');
const path = require('path');
const process = require('process');
const CleanWebpackPlugin = require('clean-webpack-plugin');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const webpack = require('webpack');


const isProd = process.env.NODE_ENV === 'production';

const distDirName = path.join('dist', 'Demo');
const demoDirName = path.join('src', 'Demo');

module.exports = {
  context: path.resolve(__dirname),

  entry: { Demo: [path.join(demoDirName, 'Demo.tsx')] },

  output: {
    publicPath: '', // Where you uploaded your bundled files. (Relative to server root)
    path: path.resolve(__dirname, distDirName), // Local disk directory to store all your output files (Absolute path)
    filename: '[name]-[hash:6].bundle.js',
  },

  module: {
    rules: [
      {
        test: /\.(j|t)sx?$/,
        exclude: [path.resolve(__dirname, 'node_modules')],
        use: 'ts-loader',
      },
      {
        test: /\.css$/,
        include: [path.resolve(__dirname, demoDirName)],
        use: ['style-loader', 'css-loader'],
      },
    ],
  },

  resolve: {
    extensions: ['.ts', '.tsx', '.js', '.jsx', '.json', '*'],
    modules: [
      path.join(__dirname, ''),
      path.join(__dirname, 'node_modules'),
    ],
  },

  plugins: [
    isProd ? new CleanWebpackPlugin(path.resolve(__dirname, distDirName)) : new webpack.HotModuleReplacementPlugin(),
    new webpack.NamedModulesPlugin(),
    new HtmlWebpackPlugin({
      filename: 'index.html',
      template: path.join(demoDirName, 'index.html'),
      inject: true,
    }),
  ],

  devServer: {
    publicPath: '/', // URL path where the webpack files are served from
    contentBase: path.join(__dirname, distDirName), // A directory to serve files non-webpack files from
    host: '0.0.0.0',
    port: process.env.PORT, // Set in docker-compose.yml
    disableHostCheck: true,
    hot: true,
    inline: true,
    watchOptions: {
      aggregateTimeout: 300,
      poll: 1000,
      ignored: /node_modules/,
    },
    historyApiFallback: true,
  },
};
/* eslint-enable import/no-commonjs */
