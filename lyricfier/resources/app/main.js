"use strict";
var Lyricfier_1 = require("./Lyricfier");
var electron = require("electron");
var Settings_1 = require("./Settings");
var app = electron.app;
var settings = new Settings_1.Settings();
var lyricfier = new Lyricfier_1.Lyricfier(app, settings, __dirname);
//# sourceMappingURL=main.js.map