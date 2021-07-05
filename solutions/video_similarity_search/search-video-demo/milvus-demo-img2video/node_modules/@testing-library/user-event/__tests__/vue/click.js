import { render, cleanup } from "@testing-library/vue";
import "@testing-library/jest-dom/extend-expect";
import userEvent from "../../src";

afterEach(cleanup);

describe("userEvent.click", () => {
  it.each(["input", "textarea"])(
    "should fire the correct events for <%s>",
    type => {
      const events = [];
      const eventsHandler = jest.fn(evt => events.push(evt.type));
      const { getByTestId } = render({
        render: function(h) {
          return h(type, {
            attrs: {
              "data-testid": "element"
            },
            on: {
              mouseover: eventsHandler,
              mousemove: eventsHandler,
              mousedown: eventsHandler,
              focus: eventsHandler,
              mouseup: eventsHandler,
              click: eventsHandler
            }
          });
        }
      });

      userEvent.click(getByTestId("element"));

      // baseElement is always <body>, si wasAnotherElementFocused is false
      expect(events).toEqual([
        "mouseover",
        "mousemove",
        "mousedown",
        "focus",
        "mouseup",
        "click"
      ]);
    }
  );

  it('should fire the correct events for <input type="checkbox">', () => {
    const events = [];
    const eventsHandler = jest.fn(evt => events.push(evt.type));

    const { getByTestId } = render({
      render: function(h) {
        return h("input", {
          attrs: {
            type: "checkbox",
            "data-testid": "element"
          },
          on: {
            mouseover: eventsHandler,
            mousemove: eventsHandler,
            mousedown: eventsHandler,
            focus: eventsHandler,
            mouseup: eventsHandler,
            click: eventsHandler,
            change: eventsHandler
          }
        });
      }
    });

    userEvent.click(getByTestId("element"));

    expect(events).toEqual([
      "mouseover",
      "mousemove",
      "mousedown",
      "mouseup",
      "click",
      "change"
    ]);

    expect(getByTestId("element")).toHaveProperty("checked", true);
  });

  it('should fire the correct events for <input type="checkbox" disabled>', () => {
    const events = [];
    const eventsHandler = jest.fn(evt => events.push(evt.type));
    const { getByTestId } = render({
      render: function(h) {
        return h("input", {
          attrs: {
            type: "checkbox",
            "data-testid": "element",
            disabled: "disabled"
          },
          on: {
            mouseover: eventsHandler,
            mousemove: eventsHandler,
            mousedown: eventsHandler,
            focus: eventsHandler,
            mouseup: eventsHandler,
            click: eventsHandler,
            change: eventsHandler
          }
        });
      }
    });

    userEvent.click(getByTestId("element"));

    expect(events).toEqual([]);

    expect(getByTestId("element")).toHaveProperty("checked", false);
  });

  it('should fire the correct events for <input type="radio">', () => {
    const events = [];
    const eventsHandler = jest.fn(evt => events.push(evt.type));
    const { getByTestId } = render({
      render: function(h) {
        return h("input", {
          attrs: {
            type: "radio",
            "data-testid": "element"
          },
          on: {
            mouseover: eventsHandler,
            mousemove: eventsHandler,
            mousedown: eventsHandler,
            focus: eventsHandler,
            mouseup: eventsHandler,
            click: eventsHandler,
            change: eventsHandler
          }
        });
      }
    });

    userEvent.click(getByTestId("element"));

    expect(events).toEqual([
      "mouseover",
      "mousemove",
      "mousedown",
      "mouseup",
      "click",
      "change"
    ]);

    expect(getByTestId("element")).toHaveProperty("checked", true);
  });

  it('should fire the correct events for <input type="radio" disabled>', () => {
    const events = [];
    const eventsHandler = jest.fn(evt => events.push(evt.type));
    const { getByTestId } = render({
      render: function(h) {
        return h("input", {
          attrs: {
            type: "radio",
            "data-testid": "element",
            disabled: "disabled"
          },
          on: {
            mouseover: eventsHandler,
            mousemove: eventsHandler,
            mousedown: eventsHandler,
            focus: eventsHandler,
            mouseup: eventsHandler,
            click: eventsHandler,
            change: eventsHandler
          }
        });
      }
    });

    userEvent.click(getByTestId("element"));

    expect(events).toEqual([]);

    expect(getByTestId("element")).toHaveProperty("checked", false);
  });

  it("should fire the correct events for <div>", () => {
    const events = [];
    const eventsHandler = jest.fn(evt => events.push(evt.type));
    const { getByTestId } = render({
      render: function(h) {
        return h("div", {
          attrs: {
            "data-testid": "div"
          },
          on: {
            mouseover: eventsHandler,
            mousemove: eventsHandler,
            mousedown: eventsHandler,
            focus: eventsHandler,
            mouseup: eventsHandler,
            click: eventsHandler
          }
        });
      }
    });

    userEvent.click(getByTestId("div"));
    expect(events).toEqual([
      "mouseover",
      "mousemove",
      "mousedown",
      "mouseup",
      "click"
    ]);
  });

  it("toggles the focus", () => {
    const { getByTestId } = render({
      template: `
      <div>
        <input data-testid="A" />
        <input data-testid="B" />
      </div>`
    });

    const a = getByTestId("A");
    const b = getByTestId("B");

    expect(a).not.toHaveFocus();
    expect(b).not.toHaveFocus();

    userEvent.click(a);
    expect(a).toHaveFocus();
    expect(b).not.toHaveFocus();

    userEvent.click(b);
    expect(a).not.toHaveFocus();
    expect(b).toHaveFocus();
  });

  it.each(["input", "textarea"])(
    "gives focus to <%s> when clicking a <label> with for",
    type => {
      const { getByTestId } = render({
        template: `
        <div>
          <label data-testid="label" for="input" />
          <component is="${type}" data-testid="input" id="input" />
        </div>`
      });

      userEvent.click(getByTestId("label"));
      expect(getByTestId("input")).toHaveFocus();
    }
  );

  it.each(["input", "textarea"])(
    "gives focus to <%s> when clicking a <label> without htmlFor",
    type => {
      const { getByTestId } = render({
        template: `
        <label data-testid="label">
          My label text
          <component is="${type}" data-testid="input" />
        </label>`
      });

      userEvent.click(getByTestId("label"));
      expect(getByTestId("input")).toHaveFocus();
    }
  );

  it.each(["input", "textarea"])(
    "gives focus to <%s> when clicking on an element contained within a <label>",
    type => {
      const { getByText, getByTestId } = render({
        template: `
        <div>
          <label for="input" data-testid="label">Label</Label>
          <component is="${type}" id="input" data-testid="input" />
        </div>`
      });

      userEvent.click(getByText("Label"));
      expect(getByTestId("input")).toHaveFocus();
    }
  );

  it('checks <input type="checkbox"> when clicking a <label> with htmlFor', () => {
    const { getByTestId } = render({
      template: `
      <div>
        <label for="input" data-testid="label">
          Label
        </label>
        <input id="input" data-testid="input" type="checkbox" />
      </div>`
    });

    expect(getByTestId("input")).toHaveProperty("checked", false);
    userEvent.click(getByTestId("label"));
    expect(getByTestId("input")).toHaveProperty("checked", true);
  });

  it('checks <input type="checkbox"> when clicking a <label> without htmlFor', () => {
    const { getByTestId } = render({
      template: `
      <div>
        <label data-testid="label">
          Label
          <input id="input" data-testid="input" type="checkbox" />
        </label>
      </div>`
    });

    expect(getByTestId("input")).toHaveProperty("checked", false);
    userEvent.click(getByTestId("label"));
    expect(getByTestId("input")).toHaveProperty("checked", true);
  });

  it("should submit a form when clicking on a <button>", () => {
    const { getByText, emitted } = render({
      template: `
      <form @submit="$emit('submit')">
        <button>Submit</button>
      </form>`
    });

    userEvent.click(getByText("Submit"));
    expect(emitted().submit).toHaveLength(1);
  });

  it('should not submit a form when clicking on a <button type="button">', () => {
    const { getByText, emitted } = render({
      template: `
      <form @submit="$emit('submit')">
        <button type="button">Submit</button>
      </form>`
    });

    userEvent.click(getByText("Submit"));
    expect(emitted()).toEqual({});
  });
});
