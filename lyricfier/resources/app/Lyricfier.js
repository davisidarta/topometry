"use strict";
var electron = require("electron");
var platform = require('os').platform();
var path = require('path');
var Lyricfier = (function () {
    function Lyricfier(app, settings, root) {
        var _this = this;
        this.rootDir = '';
        this.app = app;
        this.rootDir = root;
        this.settings = settings;
        this.subscribeSettingsEvents();
        this.loadSettingsAndApp(function () { return _this.createAppIconAndWindow(); });
    }
    Lyricfier.prototype.loadSettingsAndApp = function (cb) {
        var settingsLoaded = false;
        var appReady = false;
        this.settings.load(function () {
            settingsLoaded = true;
            if (appReady)
                cb();
        });
        this.app.on('ready', function () {
            appReady = true;
            if (settingsLoaded)
                cb();
        });
    };
    Lyricfier.prototype.getTrayIcon = function () {
        var trayImage = this.getImg('icon.png');
        // Determine appropriate icon for platform
        if (platform == 'darwin') {
            trayImage = this.getImg('tray-icon-mac.png');
        }
        else if (platform == 'win32') {
            trayImage = this.getImg('tray-icon-win.ico');
        }
        return trayImage;
    };
    Lyricfier.prototype.createAppIconAndWindow = function () {
        this.createAppIcon();
        this.createWindow();
    };
    Lyricfier.prototype.createWindow = function () {
        var _this = this;
        var options = {
            width: 500,
            height: 600,
            icon: this.getTrayIcon(),
            frame: false,
            show: false,
            alwaysOnTop: this.settings.get('alwaysOnTop')
        };
        this.window = new electron.BrowserWindow(options);
        this.window.on('close', function (e) {
            e.preventDefault();
            _this.window.hide();
        });
        this.window.loadURL(this.getView('index'));
        this.window.on('ready-to-show', function () {
            _this.window.show();
        });
        return this.window;
    };
    Lyricfier.prototype.getWindow = function () {
        return this.window;
    };
    Lyricfier.prototype.createAppIcon = function () {
        var iconPath = this.getTrayIcon();
        this.appIcon = new electron.Tray(iconPath);
        this.appIcon.setContextMenu(this.createTrayMenu());
    };
    Lyricfier.prototype.subscribeSettingsEvents = function () {
        var _this = this;
        electron.ipcMain.on('get-settings', function (event) {
            event.sender.send('settings-update', _this.settings.getRaw());
        });
        electron.ipcMain.on('settings-update', function (event, newSettings) {
            _this.settings.save(newSettings, function () { return _this.reactToSettings(); });
        });
    };
    Lyricfier.prototype.reactToSettings = function () {
        this.getWindow().setAlwaysOnTop(this.settings.get('alwaysOnTop'));
        this.getWindow().focus();
        this.appIcon.setContextMenu(this.createTrayMenu());
    };
    Lyricfier.prototype.changeSetting = function (key, value) {
        this.settings.set(key, value);
        this.reactToSettings();
        this.getWindow().webContents.send('settings-update', this.settings.getRaw());
    };
    Lyricfier.prototype.getImg = function (name) {
        return path.join(this.rootDir, 'render', 'img', name);
    };
    Lyricfier.prototype.getView = function (name) {
        return "file://" + this.rootDir + "/render/views/" + name + ".html";
    };
    Lyricfier.prototype.alwaysOnTopToggle = function () {
        this.changeSetting('alwaysOnTop', !this.settings.get('alwaysOnTop'));
    };
    Lyricfier.prototype.darkThemeToggle = function () {
        this.changeSetting('theme', this.settings.get('theme') === 'dark' ? 'light' : 'dark');
    };
    Lyricfier.prototype.createTrayMenu = function () {
        var _this = this;
        var alwaysOnTopChecked = this.settings.get('alwaysOnTop') ? '✓' : '';
        var darkTheme = this.settings.get('theme') === 'dark' ? '✓' : '';
        var menu = [
            ['Lyrics', 'showLyrics'],
            ['Dark theme ' + darkTheme, 'darkThemeToggle'],
            ['Always on top ' + alwaysOnTopChecked, 'alwaysOnTopToggle'],
            ['Open Developer Tools', 'openDeveloperTools'],
            ['Quit', 'quit']
        ];
        var template = menu.map(function (item) {
            var label = item[0], fn = item[1];
            return {
                label: label,
                click: _this[fn].bind(_this)
            };
        });
        return electron.Menu.buildFromTemplate(template);
    };
    Lyricfier.prototype.quit = function () {
        process.exit(0);
    };
    Lyricfier.prototype.showSettings = function () {
        this.getOpenWindow().webContents.send('change-view', 'Settings');
    };
    Lyricfier.prototype.showLyrics = function () {
        this.getOpenWindow().webContents.send('change-view', 'SongRender');
    };
    Lyricfier.prototype.getOpenWindow = function () {
        if (!this.window.isVisible()) {
            this.window.show();
        }
        if (this.window.isMinimized()) {
            this.window.restore();
        }
        if (!this.window.isFocused()) {
            this.window.focus();
        }
        return this.window;
    };
    Lyricfier.prototype.openDeveloperTools = function () {
        this.getOpenWindow().webContents.send('live-reload', true);
        return this.getOpenWindow().webContents.openDevTools();
    };
    return Lyricfier;
}());
exports.Lyricfier = Lyricfier;
//# sourceMappingURL=Lyricfier.js.map