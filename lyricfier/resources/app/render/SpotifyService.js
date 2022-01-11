"use strict";
var request = require('request').defaults({ timeout: 5000 });
var async = require('async');
var initialPortTest = 4370;
var SpotifyService = (function () {
    function SpotifyService() {
        this.https = false;
        this.foundPort = false;
        this.portTries = 15;
        this.albumImagesCache = {};
        this.oAuthToken = {
            t: null,
            expires: null
        };
        this.csrfToken = null;
        this.queue = [];
    }
    SpotifyService.headers = function () {
        return { 'Origin': 'https://open.spotify.com' };
    };
    SpotifyService.prototype.url = function (u) {
        var protocol = this.https ? 'https' : 'http';
        return protocol + "://127.0.0.1:" + this.port + u;
    };
    SpotifyService.prototype.getOAuthToken = function (cb) {
        var _this = this;
        if (this.oAuthToken.t) {
            return cb(null, this.oAuthToken.t);
        }
        request.get('https://open.spotify.com/token', function (err, status, body) {
            if (err) {
                return cb(err);
            }
            try {
                var json = JSON.parse(body);
                _this.oAuthToken.t = json.t;
                return cb(null, json.t);
            }
            catch (e) {
                return cb(e);
            }
        });
    };
    SpotifyService.prototype.detectPort = function (cb) {
        var _this = this;
        if (!this.foundPort) {
            this.port = initialPortTest;
        }
        async.retry(this.portTries * 2, function (finish) {
            _this.getCsrfToken(function (err) {
                if (err) {
                    console.log('FAILED WITH PORT: ', _this.port, ' and https is ', _this.https);
                    if (_this.https) {
                        _this.port++;
                        _this.https = false;
                    }
                    else {
                        _this.https = true;
                    }
                    return finish(err);
                }
                _this.foundPort = true;
                console.log('VALID PORT', _this.port);
                finish(err, _this.port);
            });
        }, cb);
    };
    SpotifyService.prototype.getCsrfToken = function (cb) {
        var _this = this;
        if (this.csrfToken) {
            return cb(null, this.csrfToken);
        }
        var url = this.url('/simplecsrf/token.json');
        request(url, {
            headers: SpotifyService.headers(),
            'rejectUnauthorized': false
        }, function (err, status, body) {
            if (err) {
                console.error('Error getting csrf token URL: ', url);
                return cb(err);
            }
            var json = JSON.parse(body);
            _this.csrfToken = json.token;
            cb(null, _this.csrfToken);
        });
    };
    SpotifyService.prototype.needsTokens = function (fn) {
        var _this = this;
        this.detectPort(function (err) {
            if (err) {
                var failDetectPort = 'No port found! Is spotify running?';
                console.error(failDetectPort, err);
                return fn(failDetectPort);
            }
            var parallelJob = {
                csrf: _this.getCsrfToken.bind(_this),
                oauth: _this.getOAuthToken.bind(_this),
            };
            async.parallel(parallelJob, fn);
        });
    };
    SpotifyService.prototype.getStatus = function (cb) {
        var _this = this;
        this.needsTokens(function (err, tokens) {
            if (err)
                return cb(err);
            var params = {
                'oauth': tokens.oauth,
                'csrf': tokens.csrf,
            };
            var url = _this.url('/remote/status.json') + '?' + _this.encodeData(params);
            request(url, {
                headers: SpotifyService.headers(),
                'rejectUnauthorized': false,
            }, function (err, status, body) {
                if (err) {
                    console.error('Error asking for status', err, ' url used: ', url);
                    return cb(err);
                }
                try {
                    var json = JSON.parse(body);
                    cb(null, json);
                }
                catch (e) {
                    var msgParseFailed = 'Status response from spotify failed';
                    console.error(msgParseFailed, ' JSON body: ', body);
                    cb(msgParseFailed, null);
                }
            });
        });
    };
    SpotifyService.prototype.getAlbumImages = function (albumUri, cb) {
        var _this = this;
        if (this.albumImagesCache[albumUri]) {
            return cb(null, this.albumImagesCache[albumUri]);
        }
        async.retry(2, function (finish) {
            var id = albumUri.split('spotify:album:')[1];
            var url = "https://api.spotify.com/v1/albums/" + id + "?oauth=" + _this.oAuthToken.t;
            request(url, function (err, status, body) {
                if (err) {
                    console.error('Error getting album images', err, ' URL: ', url);
                    return finish(err, null);
                }
                try {
                    var parsed = JSON.parse(body);
                    finish(null, parsed.images);
                    _this.albumImagesCache[albumUri] = parsed.images;
                }
                catch (e) {
                    var msgParseFail = 'Failed to parse response from spotify api';
                    console.error(msgParseFail, 'URL USED: ', url);
                    finish(msgParseFail, null);
                }
            });
        }, cb);
    };
    SpotifyService.prototype.pause = function (pause, cb) {
        var _this = this;
        this.needsTokens(function (err, tokens) {
            if (err)
                return cb(err);
            var params = {
                'oauth': tokens.oauth,
                'csrf': tokens.csrf,
                'pause': pause ? 'true' : 'false',
            };
            var url = _this.url('/remote/pause.json') + '?' + _this.encodeData(params);
            request(url, {
                headers: SpotifyService.headers(),
                'rejectUnauthorized': false,
            }, function (err, status, body) {
                if (err) {
                    return cb(err);
                }
                var json = JSON.parse(body);
                cb(null, json);
            });
        });
    };
    SpotifyService.prototype.getCurrentSong = function (cb) {
        var _this = this;
        this.getStatus(function (err, status) {
            if (err) {
                _this.foundPort = false;
                _this.csrfToken = null;
                _this.oAuthToken.t = null;
                return cb(err);
            }
            console.log('getStatus', status);
            if (status.track && status.track.track_resource) {
                var result_1 = {
                    playing: status.playing,
                    artist: status.track.artist_resource ? status.track.artist_resource.name : 'Unknown',
                    title: status.track.track_resource.name,
                    album: {
                        name: 'Unknown',
                        images: null
                    }
                };
                if (status.track.album_resource) {
                    result_1.album.name = status.track.album_resource.name;
                    return _this.getAlbumImages(status.track.album_resource.uri, function (err, images) {
                        if (!err) {
                            result_1.album.images = images;
                        }
                        return cb(null, result_1);
                    });
                }
                else {
                    return cb(null, result_1);
                }
            }
            return cb('No song', null);
        });
    };
    SpotifyService.prototype.encodeData = function (data) {
        return Object.keys(data).map(function (key) {
            return [key, data[key]].map(encodeURIComponent).join("=");
        }).join("&");
    };
    return SpotifyService;
}());
exports.SpotifyService = SpotifyService;
//# sourceMappingURL=SpotifyService.js.map