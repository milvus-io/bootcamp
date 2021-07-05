'use strict';

var expect = require('chai').expect;
var focusStore = require('./');

describe('focusStore', function() {
  beforeEach(function() {
    this.button1 = document.createElement('button');
    this.button2 = document.createElement('button');
    document.body.appendChild(this.button1);
    document.body.appendChild(this.button2);
  });

  afterEach(function() {
    this.button1.remove();
    this.button2.remove();
  });

  it('should store and restore focus', function() {
    this.button1.focus();
    focusStore.storeFocus();
    this.button2.focus();
    focusStore.restoreFocus();
    expect(document.activeElement).to.equal(this.button1);
  });

  it('should be able to clear the stored focus', function() {
    this.button1.focus();
    focusStore.storeFocus();
    this.button2.focus();
    focusStore.clearStoredFocus();
    focusStore.restoreFocus();
    expect(document.activeElement).to.equal(this.button2);
  });

  it('should not error when stored focus element has left the dom', function() {
    this.button1.focus();
    focusStore.storeFocus();
    this.button1.remove();
    expect(function() {
      focusStore.restoreFocus();
    }).not.to.throw();
  });
});
