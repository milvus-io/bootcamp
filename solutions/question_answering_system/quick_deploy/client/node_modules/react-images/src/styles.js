// @flow

// Carousel
import { containerCSS } from './components/Container';
import {
  navigationCSS,
  navigationPrevCSS,
  navigationNextCSS,
} from './components/Navigation';
import { viewCSS } from './components/View';
import {
  headerCSS,
  headerCloseCSS,
  headerFullscreenCSS,
} from './components/Header';
import {
  footerCSS,
  footerCaptionCSS,
  footerCountCSS,
} from './components/Footer';

// Modal
import {
  blanketCSS,
  dialogCSS,
  positionerCSS,
} from './components/Modal/styled';

type Props = { [key: string]: any };
type StyleDef = Props => Object;

export type GetStyles = (string, Props) => {};

export type CarouselStyles = {
  container: StyleDef,
  footer: StyleDef,
  footerCaption: StyleDef,
  footerCount: StyleDef,
  header: StyleDef,
  headerClose: StyleDef,
  headerFullscreen: StyleDef,
  navigation: StyleDef,
  navigationPrev: StyleDef,
  navigationNext: StyleDef,
  view: StyleDef,
};
export type ModalStyles = {
  blanket: StyleDef,
  dialog: StyleDef,
  positioner: StyleDef,
};
export type CarouselStylesConfig = $Shape<CarouselStyles>;
export type ModalStylesConfig = $Shape<ModalStyles>;

export const defaultCarouselStyles: CarouselStyles = {
  container: containerCSS,
  footer: footerCSS,
  footerCaption: footerCaptionCSS,
  footerCount: footerCountCSS,
  header: headerCSS,
  headerClose: headerCloseCSS,
  headerFullscreen: headerFullscreenCSS,
  navigation: navigationCSS,
  navigationPrev: navigationPrevCSS,
  navigationNext: navigationNextCSS,
  view: viewCSS,
};
export const defaultModalStyles: CarouselStyles = {
  blanket: blanketCSS,
  dialog: dialogCSS,
  positioner: positionerCSS,
};

// Merge Utility
// Allows consumers to extend a base Carousel or Modal with additional styles

export function mergeStyles(source: Object, target: Object = {}) {
  // initialize with source styles
  const styles = { ...source };

  // massage in target styles
  Object.keys(target).forEach(key => {
    if (source[key]) {
      styles[key] = (rsCss, props) => {
        return target[key](source[key](rsCss, props), props);
      };
    } else {
      styles[key] = target[key];
    }
  });

  return styles;
}
