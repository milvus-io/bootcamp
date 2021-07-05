// @flow
import React, { type ElementType } from 'react';
import { Transition } from 'react-transition-group';

const easing = 'cubic-bezier(0.23, 1, 0.32, 1)'; // easeOutQuint
const verticalOffset = 40;

type func = () => void;
type Props = {
  component: ElementType,
  in: boolean,
  innerProps: Object, // TODO
  onEntered?: func,
  onExited?: func,
};

// ==============================
// Fade
// ==============================

export const Fade = ({
  component: Tag,
  onEntered,
  onExited,
  in: inProp,
  innerProps: originalProps,
  ...props
}: Props) => {
  const enter = 300;
  const exit = 500;
  const fadeStyle = {
    transition: 'opacity 200ms',
    opacity: 0,
  };
  const fadeTransition = {
    entering: { opacity: 0 },
    entered: { opacity: 1 },
    exiting: { opacity: 0, transitionDuration: `${exit}ms` },
  };

  return (
    <Transition
      appear
      mountOnEnter
      unmountOnExit
      onEntered={onEntered}
      onExited={onExited}
      key="fade"
      in={inProp}
      timeout={{ enter, exit }}
    >
      {status => {
        const innerProps = {
          ...originalProps,
          style: {
            ...fadeStyle,
            ...fadeTransition[status],
          },
        };

        if (status === 'exited') return null;

        return <Tag innerProps={innerProps} {...props} />;
      }}
    </Transition>
  );
};

// ==============================
// Slide Up
// ==============================

export const SlideUp = ({
  component: Tag,
  onEntered,
  onExited,
  in: inProp,
  innerProps: originalProps,
  ...props
}: Props) => {
  const enter = 300;
  const exit = 500;
  const restingTransform = 'translate3d(0, 0, 0)';
  const slideStyle = {
    transition: `transform ${enter}ms ${easing}, opacity ${enter}ms ${easing}`,
    transform: restingTransform,
  };
  const slideTransition = {
    entering: {
      opacity: 0,
      transform: `translate3d(0, ${verticalOffset}px, 0) scale(0.9)`,
    },
    entered: {
      opacity: 1,
      transform: restingTransform,
    },
    exiting: {
      opacity: 0,
      transform: `translate3d(0, ${verticalOffset}px, 0) scale(0.9)`,
      transition: `transform ${exit}ms ${easing}, opacity ${exit}ms ${easing}`,
    },
  };

  return (
    <Transition
      appear
      in={inProp}
      mountOnEnter
      onEntered={onEntered}
      onExited={onExited}
      timeout={{ enter, exit }}
      unmountOnExit
    >
      {status => {
        if (status === 'exited') return null;

        const innerProps = {
          ...originalProps,
          style: {
            ...slideStyle,
            ...slideTransition[status],
          },
        };

        return <Tag innerProps={innerProps} {...props} />;
      }}
    </Transition>
  );
};
