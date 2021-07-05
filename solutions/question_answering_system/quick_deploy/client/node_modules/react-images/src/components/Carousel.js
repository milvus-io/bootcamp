// @flow
// @jsx glam
import React, { Component, type ElementRef } from 'react';
import { findDOMNode } from 'react-dom';
import glam from 'glam';
import rafScheduler from 'raf-schd';
import { ViewPager, Frame, Track, View as PageView } from 'react-view-pager';

const viewPagerStyles = {
  flex: '1 1 auto',
  position: 'relative',
};
const frameStyles = { outline: 0 };

import {
  defaultCarouselComponents,
  type CarouselComponents,
} from './defaultComponents';
import { defaultCarouselStyles, type CarouselStylesConfig } from '../styles';
import { type ModalProps } from './Modal/Modal';
import { className, isTouch } from '../utils';
import formatters from '../formatters';
import { type ViewsType } from '../types';
import componentBaseClassNames from './componentBaseClassNames';

type SpringConfig = { [key: string]: number };
export type fn = (any) => void;
export type IndicesType = Array<number>;
export type CarouselProps = {
  /* Replace any of the carousel components */
  components?: CarouselComponents,
  /* Take control of the component's view index state */
  currentIndex: number,
  // See https://github.com/souporserious/react-view-pager#frame-props
  frameProps: {
    accessibility: boolean,
    autoSize: true | false | 'width' | 'height',
    springConfig: SpringConfig,
    tag: any,
  },
  /* Formatters get called when language is used, defaults use english. */
  formatters: typeof formatters,
  /* Duration, in milliseconds, to wait before hiding controls when the user is idle */
  hideControlsWhenIdle?: number | false,
  /* Force hide or show controls on touch devices */
  showNavigationOnTouchDevice?: boolean,
  /* When envoked within a modal, props are cloned from the modal */
  modalProps?: ModalProps,
  /* Style modifier methods */
  styles: CarouselStylesConfig,
  // See https://github.com/souporserious/react-view-pager#track-props
  trackProps: {
    align: number,
    animations: Array<{ props: string, stops: Array<[number, number]> }>,
    axis: 'x' | 'y',
    contain: boolean,
    currentView: any,
    flickTimeout: number,
    infinite: boolean,
    instant: boolean,
    onRest: fn,
    onScroll: fn,
    onSwipeEnd: fn,
    onSwipeMove: fn,
    onSwipeStart: fn,
    onViewChange: (number) => void,
    springConfig: SpringConfig,
    swipe: true | false | 'mouse' | 'touch',
    swipeThreshold: number,
    tag: any,
    viewsToMove: number,
    viewsToShow: number | 'auto',
  },
  /* The items to render in the carousel */
  views: ViewsType,
};

export type CarouselState = {
  currentIndex: number,
  interactionIsIdle: boolean,
};
const defaultProps = {
  currentIndex: 0,
  formatters,
  hideControlsWhenIdle: 3000,
  showNavigationOnTouchDevice: false,
  styles: {},
  trackProps: {
    instant: !isTouch(),
    swipe: 'touch',
  },
};

const trackBaseClassName = componentBaseClassNames.Track;

class Carousel extends Component<CarouselProps, CarouselState> {
  commonProps: any; // TODO
  components: CarouselComponents;
  container: HTMLElement;
  footer: HTMLElement;
  frame: ElementRef<Frame>;
  header: HTMLElement;
  mounted: boolean = false;
  track: ElementRef<Track>;
  timer: number;

  static defaultProps = defaultProps;

  constructor(props: CarouselProps) {
    super(props);

    this.cacheComponents(props.components);

    this.state = {
      currentIndex: props.currentIndex,
      interactionIsIdle: isTouch(),
    };
  }

  componentDidMount() {
    const { hideControlsWhenIdle, modalProps } = this.props;
    const isModal = Boolean(modalProps);

    this.mounted = true;

    if (hideControlsWhenIdle && this.container) {
      this.container.addEventListener('mousedown', this.handleMouseActivity);
      this.container.addEventListener('mousemove', this.handleMouseActivity);
      this.container.addEventListener('touchmove', this.handleMouseActivity);
    }
    if (isModal) {
      this.focusViewFrame();
    }
  }
  componentDidUpdate(prevProps: CarouselProps) {
    if (prevProps.components !== this.props.components) {
      this.cacheComponents(prevProps.components);
    }

    if (this.props.currentIndex !== prevProps.currentIndex) {
      this.setState({ currentIndex: this.props.currentIndex });
    }
  }
  componentWillUnmount() {
    this.mounted = false;

    if (this.props.hideControlsWhenIdle && this.container) {
      this.container.removeEventListener('mousedown', this.handleMouseActivity);
      this.container.removeEventListener('mousemove', this.handleMouseActivity);
      this.container.removeEventListener('touchmove', this.handleMouseActivity);
      this.handleMouseActivity.cancel();
    }
  }
  cacheComponents = (comps?: CarouselComponents) => {
    this.components = defaultCarouselComponents(comps);
  };

  // ==============================
  // Refs
  // ==============================

  getContainer = (ref: HTMLElement) => {
    this.container = ref;
  };
  getFooter = (ref: HTMLElement) => {
    this.footer = ref;
  };
  getFrame = (ref: Frame) => {
    this.frame = findDOMNode(ref);
  };
  getHeader = (ref: HTMLElement) => {
    this.header = ref;
  };
  getTrack = (ref: Track) => {
    this.track = ref;
  };

  // ==============================
  // Utilities
  // ==============================

  hasPreviousView = (): boolean => {
    const { trackProps } = this.props;
    const { currentIndex } = this.state;

    return trackProps.infinite || currentIndex !== 0;
  };
  hasNextView = (): boolean => {
    const { trackProps, views } = this.props;
    const { currentIndex } = this.state;

    return trackProps.infinite || currentIndex !== views.length - 1;
  };

  getStyles = (key: string, props: {}): {} => {
    const base = defaultCarouselStyles[key](props);
    base.boxSizing = 'border-box';
    const custom = this.props.styles[key];
    return custom ? custom(base, props) : base;
  };
  // combine defaultProps with consumer props to maintain expected behaviour
  getTrackProps = (props: CarouselProps) => {
    return { ...defaultProps.trackProps, ...props.trackProps };
  };
  // combine defaultProps with consumer props to maintain expected behaviour
  getFormatters = () => {
    return { ...defaultProps.formatters, ...this.props.formatters };
  };
  getViewData = () => {
    const { views } = this.props;
    const { currentIndex } = this.state;

    return views[currentIndex];
  };
  focusViewFrame = () => {
    if (this.frame && document.activeElement !== this.frame) {
      this.frame.focus();
    }
  };

  prev = (event) => {
    event.stopPropagation();
    this.track.prev();
    this.focusViewFrame();
  };
  next = (event) => {
    event.stopPropagation();
    this.track.next();
    this.focusViewFrame();
  };

  // ==============================
  // Handlers
  // ==============================

  handleMouseActivity = rafScheduler(() => {
    clearTimeout(this.timer);

    if (this.state.interactionIsIdle) {
      this.setState({ interactionIsIdle: false });
    }

    this.timer = setTimeout(() => {
      if (this.mounted) {
        this.setState({ interactionIsIdle: true });
      }
    }, this.props.hideControlsWhenIdle);
  });
  handleViewChange = (indicies: IndicesType) => {
    const { trackProps } = this.props;

    // simplify by enforcing number
    const currentIndex = indicies[0];

    this.setState({ currentIndex });

    // call the consumer's onViewChange fn
    if (trackProps && trackProps.onViewChange) {
      trackProps.onViewChange(currentIndex);
    }
  };

  // ==============================
  // Renderers
  // ==============================

  renderNavigation = () => {
    const {
      getNextLabel,
      getPrevLabel,
      getNextTitle,
      getPrevTitle,
    } = this.getFormatters();
    const { Navigation, NavigationPrev, NavigationNext } = this.components;
    const { commonProps } = this;

    const showPrev = this.hasPreviousView();
    const showNext = this.hasNextView();
    const showNav = (showPrev || showNext) && Navigation;

    return showNav ? (
      <Navigation {...commonProps}>
        {showPrev && (
          <NavigationPrev
            {...commonProps}
            align="left"
            innerProps={{
              'aria-label': getPrevLabel(commonProps),
              onClick: this.prev,
              title: getPrevTitle(commonProps),
            }}
          />
        )}
        {showNext && (
          <NavigationNext
            {...commonProps}
            align="right"
            innerProps={{
              'aria-label': getNextLabel(commonProps),
              onClick: this.next,
              title: getNextTitle(commonProps),
            }}
          />
        )}
      </Navigation>
    ) : null;
  };
  renderFooter = () => {
    const { Footer, FooterCaption, FooterCount } = this.components;
    const { commonProps } = this;

    return Footer ? (
      <Footer
        {...commonProps}
        components={{
          Caption: FooterCaption,
          Count: FooterCount,
        }}
        innerProps={{ innerRef: this.getFooter }}
      />
    ) : null;
  };
  renderHeader = () => {
    const { Header, HeaderClose, HeaderFullscreen } = this.components;
    const { getCloseLabel, getFullscreenLabel } = this.getFormatters();
    const { commonProps } = this;

    return Header ? (
      <Header
        {...commonProps}
        getCloseLabel={getCloseLabel}
        getFullscreenLabel={getFullscreenLabel}
        components={{
          CloseButton: HeaderClose,
          FullscreenButton: HeaderFullscreen,
        }}
        data={this.getViewData()}
        innerProps={{ innerRef: this.getHeader }}
      />
    ) : null;
  };

  getCommonProps() {
    const {
      frameProps,
      trackProps,
      modalProps,
      views,
      showNavigationOnTouchDevice,
    } = this.props;
    const isModal = Boolean(modalProps);
    const isFullscreen = Boolean(modalProps && modalProps.isFullscreen);
    const { currentIndex, interactionIsIdle } = this.state;
    const currentView = this.getViewData();

    return {
      carouselProps: this.props,
      currentIndex,
      currentView,
      formatters: this.props.formatters,
      frameProps,
      getStyles: this.getStyles,
      showNavigationOnTouchDevice,
      isFullscreen,
      isModal,
      modalProps,
      interactionIsIdle,
      trackProps,
      views,
    };
  }
  render() {
    const { Container, View } = this.components;
    const { currentIndex } = this.state;
    const { frameProps, views } = this.props;
    const commonProps = (this.commonProps = this.getCommonProps());

    return (
      <Container {...commonProps} innerProps={{ innerRef: this.getContainer }}>
        {this.renderHeader()}
        <ViewPager
          tag="main"
          style={viewPagerStyles}
          className={className('pager')}
        >
          <Frame
            {...frameProps}
            ref={this.getFrame}
            className={className('frame')}
            style={frameStyles}
            tabIndex="-1"
          >
            <Track
              {...this.getTrackProps(this.props)}
              style={{ display: 'flex', alignItems: 'center' }}
              currentView={currentIndex}
              className={className(trackBaseClassName)}
              onViewChange={this.handleViewChange}
              ref={this.getTrack}
            >
              {views &&
                views.map((data, index) => {
                  return (
                    <PageView className={className('view-wrapper')} key={index}>
                      <View {...commonProps} data={data} index={index} />
                    </PageView>
                  );
                })}
            </Track>
          </Frame>
          {this.renderNavigation()}
        </ViewPager>
        {this.renderFooter()}
      </Container>
    );
  }
}
export default Carousel;
export type CarouselType = typeof Carousel;
