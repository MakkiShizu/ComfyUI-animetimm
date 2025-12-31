import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";
const TARGET_NODES = new Set(["AnimeTimmNode"]);

app.registerExtension({
  name: "ComfyUI-animetimm.INFO",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (TARGET_NODES.has(nodeData.name)) {
      const INFO_VALUE = Symbol("infoValue");

      const onExecuted = nodeType.prototype.onExecuted;

      nodeType.prototype.onExecuted = function (message) {
        const r = onExecuted?.apply?.(this, arguments);

        const pos = this.widgets.findIndex((w) => w.name === "info");

        if (pos !== -1) {
          for (let i = this.widgets.length - 1; i >= pos; i--) {
            this.widgets[i].onRemove?.();
            this.widgets.splice(i, 1);
          }
        }

        if (message.info) {
          if (Array.isArray(message.info)) {
            for (const item of message.info) {
              const w = ComfyWidgets["STRING"](
                this,
                "info",
                [
                  "STRING",
                  {
                    multiline: true,
                  },
                ],
                app
              ).widget;

              w.inputEl.readOnly = true;
              w.inputEl.style.opacity = 0.8;
              w.value =
                typeof item === "object"
                  ? JSON.stringify(item, null, 2)
                  : item.toString();
            }
          } else {
            let infoText;
            if (typeof message.info === "object") {
              infoText = JSON.stringify(message.info, null, 2);
            } else {
              infoText = message.info.toString();
            }

            this[INFO_VALUE] = infoText;

            const w = ComfyWidgets["STRING"](
              this,
              "info",
              [
                "STRING",
                {
                  multiline: true,
                },
              ],
              app
            ).widget;

            w.inputEl.readOnly = true;
            w.inputEl.style.opacity = 0.8;
            w.value = infoText;
          }
        }

        this.onResize?.(this.size);

        return r;
      };

      const configure = nodeType.prototype.configure;
      nodeType.prototype.configure = function () {
        const config = arguments[0];
        if (this[INFO_VALUE] && config) {
          if (!config.widgets_values) {
            config.widgets_values = [];
          }
          config.widgets_values.push(this[INFO_VALUE]);
        }
        return configure?.apply(this, arguments);
      };

      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        onConfigure?.apply(this, arguments);

        if (this.widgets_values?.length) {
          const savedInfoValue =
            this.widgets_values[this.widgets_values.length - 1];

          requestAnimationFrame(() => {
            if (savedInfoValue) {
              const pos = this.widgets.findIndex((w) => w.name === "info");
              if (pos !== -1) {
                for (let i = this.widgets.length - 1; i >= pos; i--) {
                  this.widgets[i].onRemove?.();
                  this.widgets.splice(i, 1);
                }
              }

              const w = ComfyWidgets["STRING"](
                this,
                "info",
                [
                  "STRING",
                  {
                    multiline: true,
                  },
                ],
                app
              ).widget;

              w.inputEl.readOnly = true;
              w.inputEl.style.opacity = 0.8;
              w.value = savedInfoValue;

              this[INFO_VALUE] = savedInfoValue;

              this.onResize?.(this.size);
            }
          });
        }
      };
    }
  },
});
