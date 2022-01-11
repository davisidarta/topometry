"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var vue_class_component_1 = require("vue-class-component");
var template_1 = require("./template");
var SettingsRender = (function () {
    function SettingsRender() {
    }
    SettingsRender.prototype.openExternal = function (url) {
        this.shell.openExternal(url);
    };
    SettingsRender.prototype.setTheme = function (theme) {
        this.settings.theme = theme;
    };
    SettingsRender.prototype.goBack = function () {
        this.$parent.changeView('SongRender');
    };
    return SettingsRender;
}());
SettingsRender = __decorate([
    vue_class_component_1.default({
        props: {
            'ipc': {
                'type': Object
            },
            'shell': {
                'type': Object
            },
            'settings': {
                'type': Object
            },
            'onChangeSettings': {
                'type': Function
            }
        },
        template: template_1.template('Settings')
    })
], SettingsRender);
exports.SettingsRender = SettingsRender;
//# sourceMappingURL=SettingsRender.js.map