// @flow

export function getSource({ data, isFullscreen }) {
  let { source = data.src } = data;
  if (typeof source === 'string') return source;

  return isFullscreen ? source.fullscreen : source.regular;
}

export function getThumbnail({ data }) {
  const { source } = data;

  if (typeof source === 'string') return source;

  return source.thumbnail;
}
