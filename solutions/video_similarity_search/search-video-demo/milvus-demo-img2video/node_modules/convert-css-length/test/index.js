/*
 * decaffeinate suggestions:
 * DS102: Remove unnecessary code created because of implicit returns
 * Full docs: https://github.com/decaffeinate/decaffeinate/blob/master/docs/suggestions.md
 */
import { expect } from 'chai';
import convertCSSLength from '../src/index';

const convertLength = convertCSSLength('16px');

describe('convert-css-length', function() {
  it('should exist', () => expect(convertLength).to.exist());

  it('should handle cases where from and to units are the same', function() {
    expect(convertLength('1em', 'em')).to.equal('1em');
    expect(convertLength('1rem', 'rem')).to.equal('1rem');
    return expect(convertLength('1px', 'px')).to.equal('1px');
  });

  it('should convert em to px', function() {
    expect(convertLength('1em', 'px')).to.equal('16px');
    expect(convertLength('.5em', 'px')).to.equal('8px');
    return expect(convertLength('1.5em', 'px')).to.equal('24px');
  });

  it('should convert em to rem', function() {
    expect(convertLength('1em', 'rem')).to.equal('1rem');
    expect(convertLength('.5em', 'rem')).to.equal('0.5rem');
    return expect(convertLength('1.5em', 'rem')).to.equal('1.5rem');
  });

  it('should convert ex to px', function() {
    expect(convertLength('1ex', 'px')).to.equal('32px');
    expect(convertLength('.5ex', 'px')).to.equal('16px');
    return expect(convertLength('1.5ex', 'px')).to.equal('48px');
  });

  it('should convert rem to px', function() {
    expect(convertLength('1rem', 'px')).to.equal('16px');
    expect(convertLength('.5rem', 'px')).to.equal('8px');
    return expect(convertLength('1.5rem', 'px')).to.equal('24px');
  });

  it('should convert px to em', function() {
    expect(convertLength('16px', 'em')).to.equal('1em');
    expect(convertLength('8px', 'em')).to.equal('0.5em');
    return expect(convertLength('24px', 'em')).to.equal('1.5em');
  });

  it('should convert px to ex', function() {
    expect(convertLength('16px', 'ex')).to.equal('0.5ex');
    expect(convertLength('8px', 'ex')).to.equal('0.25ex');
    return expect(convertLength('24px', 'ex')).to.equal('0.75ex');
  });

  it('should convert px to rem', function() {
    expect(convertLength('16px', 'rem')).to.equal('1rem');
    expect(convertLength('8px', 'rem')).to.equal('0.5rem');
    return expect(convertLength('24px', 'rem')).to.equal('1.5rem');
  });

  // With context.
  it('should convert em to px with fromContext', function() {
    expect(convertLength('1em', 'px', '14px')).to.equal('14px');
    expect(convertLength('.5em', 'px', '14px')).to.equal('7px');
    return expect(convertLength('1.5em', 'px', '14px')).to.equal('21px');
  });

  it('should convert em to rem with fromContext', function() {
    expect(convertLength('1em', 'rem', '14px')).to.equal('0.875rem');
    expect(convertLength('.5em', 'rem', '14px')).to.equal('0.4375rem');
    return expect(convertLength('1.5em', 'rem', '14px')).to.equal('1.3125rem');
  });

  return it('should convert px to em with toContext', function() {
    expect(convertLength('16px', 'em', null, '14px')).to.equal('1.14286em');
    expect(convertLength('8px', 'em', null, '14px')).to.equal('0.57143em');
    return expect(convertLength('24px', 'em', null, '14px')).to.equal('1.71429em');
  });
});
