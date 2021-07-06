import React, { PureComponent } from 'react'

class GitHubButton extends PureComponent {
  constructor (props) {
    super(props)
    this.$ = React.createRef()
    this._ = React.createRef()
  }
  render () {
    return React.createElement('span', { ref: this.$ }, React.createElement('a', { ...this.props, ref: this._ }, this.props.children))
  }
  componentDidMount () {
    this.paint()
  }
  getSnapshotBeforeUpdate () {
    this.reset()
    return null
  }
  componentDidUpdate () {
    this.paint()
  }
  componentWillUnmount () {
    this.reset()
  }
  paint () {
    const _ = this.$.current.appendChild(document.createElement('span'))
    import(/* webpackMode: "eager" */ 'github-buttons').then(({ render }) => {
      render(_.appendChild(this._.current), function (el) {
        try {
          _.parentNode.replaceChild(el, _)
        } catch (_) {}
      })
    })
  }
  reset () {
    this.$.current.replaceChild(this._.current, this.$.current.lastChild)
  }
}

export default GitHubButton
