import React, { Component } from "react";

export default class FadeIn extends Component {
  constructor() {
    super();
    this.state = {
      maxIsVisible: 0
    };
  }

  get delay() {
    return this.props.delay || 50;
  }

  get transitionDuration() {
    return this.props.transitionDuration || 400;
  }

  componentDidMount() {
    const count = React.Children.count(this.props.children);
    let i = 0;
    this.interval = setInterval(() => {
      i++;
      if (i > count) clearInterval(this.interval);

      this.setState({ maxIsVisible: i });
    }, this.delay);
  }

  componentWillUnmount() {
    clearInterval(this.interval);
  }

  render() {
    const transitionDuration = this.transitionDuration;
    return (
      <div className={this.props.className}>
        {React.Children.map(this.props.children, (child, i) => {
          return (
            <div
              className={this.props.childClassName}
              style={{
                transition: `opacity ${transitionDuration}ms, top ${transitionDuration}ms`,
                position: "relative",
                top: this.state.maxIsVisible > i ? 0 : 20,
                opacity: this.state.maxIsVisible > i ? 1 : 0
              }}
            >
              {child}
            </div>
          );
        })}
      </div>
    );
  }
}
