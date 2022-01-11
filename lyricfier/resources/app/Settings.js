"use strict";
var storage = require("electron-json-storage");
exports.defaultSettings = {
    alwaysOnTop: false,
    theme: 'light',
    fontSize: 'twelve-pt',
    refreshInterval: 5000
};
var Settings = (function () {
    function Settings() {
        this.raw = {};
    }
    Settings.prototype.getRaw = function () {
        return this.raw;
    };
    Settings.prototype.setDefaults = function (settings) {
        for (var _i = 0, _a = Object.keys(exports.defaultSettings); _i < _a.length; _i++) {
            var k = _a[_i];
            settings[k] = exports.defaultSettings[k];
        }
    };
    Settings.prototype.load = function (ready) {
        var _this = this;
        this.setDefaults(this.raw);
        storage.get('settings', function (err, savedSettings) {
            if (err)
                savedSettings = {};
            for (var attr in _this.raw) {
                if ((attr in savedSettings) === false) {
                    savedSettings[attr] = _this.raw[attr];
                }
            }
            _this.raw = savedSettings;
            ready();
        });
    };
    Settings.prototype.save = function (newSettings, ready) {
        var oldSettings = {};
        for (var attr in newSettings) {
            oldSettings[attr] = this.raw[attr];
        }
        if (JSON.stringify(newSettings) === JSON.stringify(oldSettings)) {
            console.log('no modifications');
        }
        else {
            console.log('modified settings!');
            for (var attr in newSettings) {
                this.raw[attr] = newSettings[attr];
            }
            this.persist();
        }
        ready && ready();
    };
    Settings.prototype.persist = function (ready) {
        storage.set('settings', this.raw, function (err) {
            if (err)
                console.log('Err persisting settings', err);
            ready && ready(err);
        });
    };
    Settings.prototype.set = function (key, value) {
        this.raw[key] = value;
        this.persist();
    };
    Settings.prototype.get = function (key) {
        return this.raw[key];
    };
    return Settings;
}());
exports.Settings = Settings;
//# sourceMappingURL=Settings.js.map