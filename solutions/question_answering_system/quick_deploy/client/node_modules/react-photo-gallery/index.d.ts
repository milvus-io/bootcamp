import * as React from 'react'

/**
 * Photos array item properties (passed into Gallery's photos property)
 */
export type PhotoProps<CustomPhotoProps extends object = {}> = {
  /**
   * the img src attribute value of the image
   */
  src: string
  /**
   * srcSet attribute of the image
   */
  srcSet?: string | string[]
  /**
   * sizes attribute of the image
   */
  sizes?: string | string[]
  /**
   *  original width of the gallery image (only used for calculating aspect ratio)
   */
  width: number
  /**
   *  original height of the gallery image (only used for calculating aspect ratio)
   */
  height: number
  /**
   *  alt text of the gallery image
   */
  alt?: string
  /**
   * key to be used on component
   */
  key?: string
} & CustomPhotoProps

export type renderImageClickHandler = (
  event: React.MouseEvent,
  photo: object & {
    index: number
  },
) => void

/**
 * If you're passing a function component to renderImage you will receive back these props:
 */
export interface RenderImageProps<CustomPhotoProps extends object = {}> {
  /**
   * margin prop optionally passed into Gallery by user
   */
  margin?: string
  /**
   * the index of the photo within the Gallery
   */
  index: number
  /**
   * the individual object passed into Gallery's
   * photos array prop, with all the same props except recalculated height and width
   */
  photo: PhotoProps<CustomPhotoProps>

  onClick: renderImageClickHandler | null
  direction: 'row' | 'column'
  top?: number
  left?: number
}

export type PhotoClickHandler<CustomPhotoProps extends object = {}> = (
  event: React.MouseEvent,
  photos: {
    index: number
    next: PhotoProps<CustomPhotoProps> | null
    photo: PhotoProps<CustomPhotoProps>
    previous: PhotoProps<CustomPhotoProps> | null
  },
) => void

export interface GalleryProps<CustomPhotoProps extends object = {}> {
  photos: Array<PhotoProps<CustomPhotoProps>>
  /**
   * applies to column layouts only (direction=column)
   * number of columns or a function which receives the container width
   * and should return the desired number of columns; defaults to Gallery's breakpoint choosing
   */
  columns?: number | ((containerWidth: number) => number)
  /**
   * applies to row layouts only (direction=row)
   * the ideal height of each row or a function which receives the container width
   * and should return the desired ideal height for each row; defaults to 300px
   */
  targetRowHeight?: number | ((containerWidth: number) => number)
  /**
   * applies to row layouts only (direction=row)
   * the maximum amount of neighboring nodes to measure per current node visiting
   * don't change unless you understand the algorithm, see docs
   * defaults to a couple breakpoints
   */
  limitNodeSearch?: number | ((containerWidth: number) => number)
  /**
   * do something when the user clicks a photo;
   * receives arguments event and an object containing the index,
   * photo obj originally sent and the next and previous photos in the gallery if they exist
   */
  onClick?: PhotoClickHandler<CustomPhotoProps>

  /**
   * number of margin pixels around each entire image
   */
  margin?: number
  /**
   * column or row based layout
   */
  direction?: string

  renderImage?: React.ComponentType<RenderImageProps<CustomPhotoProps>>
}

export type GalleryI<
  CustomPhotoProps extends object = {}
> = React.ComponentClass<GalleryProps<CustomPhotoProps>>

declare const Gallery: GalleryI

export default Gallery

