"use strict";
var __extends = (this && this.__extends) || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
};
var SearchLyrics_1 = require("./SearchLyrics");
var cheerio = require("cheerio");
var Genius = (function (_super) {
    __extends(Genius, _super);
    function Genius() {
        var _this = _super.apply(this, arguments) || this;
        _this.name = 'Genius';
        return _this;
    }
    Genius.prototype.search = function (title, artist, cb) {
        var _this = this;
        var url = "http://genius.com/search?q=" + encodeURIComponent(artist) + " " + encodeURIComponent(title);
        this.doReq(url, function (err, res, body) {
            if (err || res.statusCode != 200) {
                return cb('Error response searching genius');
            }
            try {
                var songUrl = /href="(.*)"\s*class="\s*song_link\s*"/.exec(body)[1];
                _this.getSong(songUrl, cb);
            }
            catch (e) {
                cb('Genius fail');
            }
        });
    };
    Genius.prototype.getSong = function (url, cb) {
        var _this = this;
        this.doReq(url, function (err, res, body) {
            if (err || res.statusCode != 200) {
                return cb("Error response getting song from Genius");
            }
            try {
                var lyric = _this.parseContent(body);
                return cb(null, { lyric: lyric, url: url });
            }
            catch (e) {
                cb("Genius fail parsing lyrics");
            }
        });
    };
    Genius.prototype.parseContent = function (body) {
        var txt = cheerio.load(body)('.lyrics p').text();
        var el = document.createElement("textarea");
        el.innerHTML = txt;
        return el.value;
    };
    return Genius;
}(SearchLyrics_1.SearchLyrics));
exports.Genius = Genius;
//# sourceMappingURL=Genius.js.map