/// <reference path="./render-typings.d.ts" />
"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var vue_class_component_1 = require("vue-class-component");
var electron_1 = require("electron");
var SettingsRender_1 = require("./SettingsRender");
var SongRender_1 = require("./SongRender");
var template_1 = require("./template");
var Settings_1 = require("../Settings");
var toastr = require('toastr');
toastr.options.positionClass = 'toast-bottom-left';
toastr.options.timeout = '60000';
var LyricfierRender = (function () {
    function LyricfierRender() {
        this.liveReload = false;
        this.lastMessageTime = 0;
        this.lastMessage = '';
    }
    LyricfierRender.prototype.listenStatus = function (msg) {
        if (msg === this.lastMessage) {
            // not spamming same message.
            var now = new Date();
            var last = new Date(this.lastMessageTime);
            last.setMilliseconds(last.getMilliseconds() + 4500);
            if (now < last) {
                return;
            }
        }
        this.lastMessageTime = (new Date()).getTime();
        this.lastMessage = msg;
        toastr.info(msg);
    };
    LyricfierRender.prototype.data = function () {
        return {
            menu: [
                'SongRender',
                'SettingsRender'
            ],
            ipc: electron_1.ipcRenderer,
            shell: electron_1.shell,
            liveReload: false,
            currentView: 'SongRender',
            settings: Settings_1.defaultSettings,
        };
    };
    LyricfierRender.prototype.ready = function () {
        var _this = this;
        this.ipc.send('get-settings');
        console.log('setting update setup');
        this.ipc.on('settings-update', function (event, arg) {
            _this.settings = arg;
            console.log(arg);
        });
        this.ipc.on('change-view', function (event, page) {
            _this.changeView(page);
        });
        this.ipc.on('live-reload', function (event, status) {
            _this.liveReload = status;
        });
        this.ipc.on('status', function (event, msg) {
            _this.listenStatus(msg);
        });
    };
    LyricfierRender.prototype.saveSettings = function () {
        this.ipc.send('settings-update', JSON.parse(JSON.stringify(this.settings)));
    };
    LyricfierRender.prototype.switchView = function () {
        if (this.isView('SongRender')) {
            this.changeView('SettingsRender');
        }
        else {
            this.changeView('SongRender');
        }
    };
    LyricfierRender.prototype.changeView = function (page) {
        this.currentView = page;
    };
    LyricfierRender.prototype.isView = function (page) {
        return this.currentView === page;
    };
    return LyricfierRender;
}());
LyricfierRender = __decorate([
    vue_class_component_1.default({
        components: {
            SettingsRender: SettingsRender_1.SettingsRender,
            SongRender: SongRender_1.SongRender
        },
        template: template_1.template('Lyricfier')
    })
], LyricfierRender);
module.exports = LyricfierRender;
//# sourceMappingURL=LyricfierRender.js.map