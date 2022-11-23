// import { CSSTransitionGroup } from 'react-transition-group' // ES6
//实现css动画
import React from "react";
export default class TodoList extends React.Component {
  constructor(props) {
    super(props);
    this.state = { items: ['hello', 'world', 'click', 'me'] };
    this.handleAdd = this.handleAdd.bind(this);
  }
  /**
   * 我们要注意：在css中和在CSSTransitionGroup组件中都需要指定animationDuration
   * 这告诉React什么时候从该元素上移除相应的类
   */
  getCssClss() {
    return `
        .example-enter {
          opacity: 0.01;
        }

        .example-enter.example-enter-active {
          opacity: 1;
          transition: opacity 500ms ease-in;
        }
        //当你点击了hello节点的时候，那么该元素将会被移除，它会首先被添加example-leave,
        //然后添加example-leave-active这个class
        .example-leave {
          opacity: 1;
        }
        .example-leave.example-leave-active {
          opacity: 0.01;
          transition: opacity 300ms ease-in;
        }
        .example-appear {
          opacity: 0.01;
        }
        .example-appear.example-appear-active {
          opacity: 1;
          transition: opacity .5s ease-in;
        }
    `
  }
  /**
   * 添加一个元素
   */
  handleAdd() {
    const newItems = this.state.items.concat([
      prompt('Enter some text')
    ]);
    this.setState({ items: newItems });
  }

  /**
   * 移除某一个元素。然后导致我们的组件reRender,对于我们要移除的那个DOM，我们会
   * 首先添加example-enter和example-enter-active。这是通过我们的key来判断的
   */
  handleRemove(i) {
    let newItems = this.state.items.slice();
    newItems.splice(i, 1);
    this.setState({ items: newItems });
  }

  /**
   * 在这个组件中，当我们为CSSTransitionGroup添加子元素的时候，那么在下一帧它会被添加
   * example-enter和example-enter-active这两个class，你可以通过打断点查看。这是基于我们
   * 指定的transitionName属性来判断的。
   */
  render() {
    const items = this.state.items.map((item, i) => (
      <div key={item} onClick={() => this.handleRemove(i)}>
        {item}
      </div>
    ));

    return (
      <div>
        <style dangerouslySetInnerHTML={{ __html: this.getCssClss() }} />
        <button onClick={this.handleAdd}>Add Item</button>
        {/* <CSSTransitionGroup
          transitionName="example"
          transitionEnterTimeout={500}
          transitionLeaveTimeout={300}
          transitionAppear={true}
          transitionAppearTimeout={500}>
          {items}
        </CSSTransitionGroup> */}
      </div>
    );
  }
}