// @flow

import Container from './Container';
import Footer, { FooterCaption, FooterCount } from './Footer';
import Header, { HeaderClose, HeaderFullscreen } from './Header';
import { Navigation, NavigationPrev, NavigationNext } from './Navigation';
import { Blanket, Dialog, Positioner } from './Modal/styled';
import View from './View';

export type CarouselComponents = {
  Container: typeof Container,
  Footer: typeof Footer,
  FooterCaption: typeof FooterCaption,
  FooterCount: typeof FooterCount,
  Header: typeof Header,
  HeaderClose: typeof HeaderClose,
  HeaderFullscreen: typeof HeaderFullscreen,
  Navigation: typeof Navigation,
  NavigationPrev: typeof NavigationPrev,
  NavigationNext: typeof NavigationNext,
  View: typeof View,
};

export const carouselComponents: CarouselComponents = {
  Container: Container,
  Footer: Footer,
  FooterCaption: FooterCaption,
  FooterCount: FooterCount,
  Header: Header,
  HeaderClose: HeaderClose,
  HeaderFullscreen: HeaderFullscreen,
  Navigation: Navigation,
  NavigationPrev: NavigationPrev,
  NavigationNext: NavigationNext,
  View: View,
};

export const defaultCarouselComponents = (
  providedComponents?: CarouselComponents
) => ({
  ...carouselComponents,
  ...providedComponents,
});

// ==============================
// Modal
// ==============================

export type ModalComponents = {
  Blanket: typeof Blanket,
  Positioner: typeof Positioner,
  Dialog: typeof Dialog,
};

export const modalComponents: ModalComponents = {
  Blanket: Blanket,
  Positioner: Positioner,
  Dialog: Dialog,
};

export const defaultModalComponents = (
  providedComponents?: ModalComponents
) => ({
  ...modalComponents,
  ...providedComponents,
});
