"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var vue_class_component_1 = require("vue-class-component");
var Searcher_1 = require("./Searcher");
var template_1 = require("./template");
var SpotifyService_1 = require("./SpotifyService");
var SongRender = (function () {
    function SongRender() {
        this.timer = null;
    }
    SongRender.prototype.data = function () {
        return {
            song: null,
            lastSongSync: {},
            searcher: new Searcher_1.Searcher(),
        };
    };
    SongRender.prototype.scheduleNextCall = function () {
        var _this = this;
        if (this.timer) {
            clearTimeout(this.timer);
        }
        console.warn('Scheduling ', this.settings.refreshInterval / 1000, ' seconds');
        this.timer = setTimeout(function () {
            _this.refresh();
        }, this.settings.refreshInterval);
    };
    SongRender.prototype.ready = function () {
        this.refresh();
    };
    SongRender.prototype.refresh = function () {
        var _this = this;
        console.log('refreshing');
        this.getSpotify().getCurrentSong(function (err, song) {
            if (err) {
                _this.showError('Current song error: ' + err);
                _this.scheduleNextCall();
            }
            else if (_this.isLastSong(song)) {
                console.log('is last song nothing to do here');
                _this.scheduleNextCall();
            }
            else {
                console.log('is not last song searching by title and artist');
                song.lyric = 'Loading Lyrics...';
                _this.displaySong(song);
                _this.saveLastSong(song);
                _this.searcher.search(song.title, song.artist, function (err, result) {
                    if (err) {
                        _this.showError('Plugin error: ' + err);
                        return;
                    }
                    if (result.lyric === null) {
                        song.lyric = 'Sorry, couldn\'t find lyrics for this song!';
                        song.sourceUrl = null;
                        song.sourceName = null;
                    }
                    else {
                        song.lyric = result.lyric;
                        song.sourceUrl = result.sourceUrl;
                        song.sourceName = result.sourceName;
                    }
                    _this.displaySong(song);
                    _this['$nextTick'](function () {
                        document.getElementById("lyricBox").scrollTop = 0;
                    });
                    _this.scheduleNextCall();
                });
            }
        });
    };
    SongRender.prototype.displaySong = function (song) {
        var newSongObject = {};
        for (var _i = 0, _a = Object.keys(song); _i < _a.length; _i++) {
            var k = _a[_i];
            newSongObject[k] = song[k];
        }
        this.song = newSongObject;
    };
    SongRender.prototype.isLastSong = function (song) {
        for (var _i = 0, _a = ['artist', 'title']; _i < _a.length; _i++) {
            var k = _a[_i];
            if (song[k] !== this.lastSongSync[k]) {
                return false;
            }
        }
        return true;
    };
    SongRender.prototype.saveLastSong = function (song) {
        for (var _i = 0, _a = Object.keys(song); _i < _a.length; _i++) {
            var k = _a[_i];
            this.lastSongSync[k] = song[k];
        }
    };
    SongRender.prototype.getSpotify = function () {
        if (!this.service) {
            this.service = new SpotifyService_1.SpotifyService();
        }
        return this.service;
    };
    SongRender.prototype.openExternal = function (url) {
        this.shell.openExternal(url);
    };
    return SongRender;
}());
SongRender = __decorate([
    vue_class_component_1.default({
        props: {
            'shell': {
                'type': Object
            },
            'showError': {
                'type': Function
            },
            'settings': {
                'type': Object
            }
        },
        template: template_1.template('Song')
    })
], SongRender);
exports.SongRender = SongRender;
//# sourceMappingURL=SongRender.js.map