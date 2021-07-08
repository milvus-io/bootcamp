import { render, cleanup } from "@testing-library/vue";
import "@testing-library/jest-dom/extend-expect";
import userEvent from "../../src";

afterEach(cleanup);

describe("userEvent.selectOptions", () => {
  it.each(["select", "select multiple"])(
    "should fire the correct events for <%s>",
    type => {
      const events = [];
      const eventsHandler = jest.fn(evt => events.push(evt.type));
      const multiple = type === "select multiple";
      const eventHandlers = {
        mouseover: eventsHandler,
        mousemove: eventsHandler,
        mousedown: eventsHandler,
        focus: eventsHandler,
        mouseup: eventsHandler,
        click: eventsHandler
      };

      const { getByTestId } = render({
        render: function(h) {
          return h(
            "select",
            {
              attrs: {
                "data-testid": "element",
                ...(multiple && { multiple: true })
              },
              on: eventHandlers
            },
            [
              h("option", { attrs: { value: "1" } }, "1"),
              h("option", { attrs: { value: "2" } }, "2"),
              h("option", { attrs: { value: "3" } }, "3")
            ]
          );
        }
      });

      userEvent.selectOptions(getByTestId("element"), "1");

      expect(events).toEqual([
        "mouseover",
        "mousemove",
        "mousedown",
        "focus",
        "mouseup",
        "click",
        "mouseover", // The events repeat because we click on the child OPTION too
        "mousemove", // But these specifically are the events bubbling up to the <select>
        "mousedown",
        // "focus", // Focus event isn't being emitted?
        "mouseup",
        "click"
      ]);
    }
  );

  it("should fire the correct events on selected OPTION child with <select>", () => {
    function handleEvent(evt) {
      const optValue = parseInt(evt.target.value);
      events[optValue] = [...(events[optValue] || []), evt.type];
    }

    const events = [];
    const eventsHandler = jest.fn(handleEvent);
    const eventHandlers = {
      mouseover: eventsHandler,
      mousemove: eventsHandler,
      mousedown: eventsHandler,
      focus: eventsHandler,
      mouseup: eventsHandler,
      click: eventsHandler
    };

    const { getByTestId } = render({
      render: function(h) {
        return h(
          "select",
          {
            attrs: {
              "data-testid": "element"
            }
          },
          [
            h("option", { attrs: { value: "1" }, on: eventHandlers }, "1"),
            h("option", { attrs: { value: "2" }, on: eventHandlers }, "2"),
            h("option", { attrs: { value: "3" }, on: eventHandlers }, "3")
          ]
        );
      }
    });

    userEvent.selectOptions(getByTestId("element"), ["2"]);

    expect(events[1]).toBe(undefined);
    expect(events[3]).toBe(undefined);
    expect(events[2]).toEqual([
      "mouseover",
      "mousemove",
      "mousedown",
      "focus",
      "mouseup",
      "click"
    ]);
  });

  it("should fire the correct events on selected OPTION children with <select multiple>", () => {
    function handleEvent(evt) {
      const optValue = parseInt(evt.target.value);
      events[optValue] = [...(events[optValue] || []), evt.type];
    }

    const events = [];
    const eventsHandler = jest.fn(handleEvent);
    const eventHandlers = {
      mouseover: eventsHandler,
      mousemove: eventsHandler,
      mousedown: eventsHandler,
      focus: eventsHandler,
      mouseup: eventsHandler,
      click: eventsHandler
    };

    const { getByTestId } = render({
      render: function(h) {
        return h(
          "select",
          {
            attrs: {
              "data-testid": "element",
              multiple: true
            }
          },
          [
            h("option", { attrs: { value: "1" }, on: eventHandlers }, "1"),
            h("option", { attrs: { value: "2" }, on: eventHandlers }, "2"),
            h("option", { attrs: { value: "3" }, on: eventHandlers }, "3")
          ]
        );
      }
    });

    userEvent.selectOptions(getByTestId("element"), ["1", "3"]);

    expect(events[2]).toBe(undefined);
    expect(events[1]).toEqual([
      "mouseover",
      "mousemove",
      "mousedown",
      "focus",
      "mouseup",
      "click"
    ]);

    expect(events[3]).toEqual([
      "mouseover",
      "mousemove",
      "mousedown",
      "focus",
      "mouseup",
      "click"
    ]);
  });

  it("sets the selected prop on the selected OPTION", () => {
    const { getByTestId } = render({
      template: `
        <form>
          <select data-testid="element" multiple>
            <option value="1" data-testid="val1">1</option>
            <option value="2" data-testid="val2">2</option>
            <option value="3" data-testid="val3">3</option>
          </select>
        </form>`
    });

    userEvent.selectOptions(getByTestId("element"), ["1", "3"]);

    expect(getByTestId("val1").selected).toBe(true);
    expect(getByTestId("val2").selected).toBe(false);
    expect(getByTestId("val3").selected).toBe(true);
  });

  it("sets the selected prop on the selected OPTION using htmlFor", () => {
    const { getByTestId } = render({
      template: `
        <form>
          <label htmlFor="select">Example Select</label>
          <select id="select" data-testid="element">
            <option data-testid="val1" value="1">1</option>
            <option data-testid="val2" value="2">2</option>
            <option data-testid="val3" value="3">3</option>
          </select>
        </form>`
    });

    userEvent.selectOptions(getByTestId("element"), "2");

    expect(getByTestId("val1").selected).toBe(false);
    expect(getByTestId("val2").selected).toBe(true);
    expect(getByTestId("val3").selected).toBe(false);
  });

  it("sets the selected prop on the selected OPTION using nested SELECT", () => {
    const { getByTestId } = render({
      template: `
        <form>
          <label>
            Example Select
            <select data-testid="element">
              <option data-testid="val1" value="1">1</option>
              <option data-testid="val2" value="2">2</option>
              <option data-testid="val3" value="3">3</option>
            </select>
          </label>
        </form>`
    });

    userEvent.selectOptions(getByTestId("element"), "2");

    expect(getByTestId("val1").selected).toBe(false);
    expect(getByTestId("val2").selected).toBe(true);
    expect(getByTestId("val3").selected).toBe(false);
  });
});
