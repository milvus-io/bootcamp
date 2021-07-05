// This file exists as an entry point for bundling our umd builds.
// Both in rollup and in webpack, umd builds built from es6 modules are not
// compatible with mixed imports (which exist in index.js)
// This file does away with named imports in favor of a single export default.

import Carousel from './components/Carousel';
import ModalGateway from './components/Modal/Gateway';
import Modal from './components/Modal/Modal';
import {
  carouselComponents,
  modalComponents,
} from './components/defaultComponents';

Carousel.ModalGateway = ModalGateway;
Carousel.Modal = Modal;
Carousel.carouselComponents = carouselComponents;
Carousel.modalComponents = modalComponents;
export default Carousel;
