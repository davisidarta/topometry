"use strict";
var __extends = (this && this.__extends) || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
};
var SearchLyrics_1 = require("./SearchLyrics");
var MusicMatch = (function (_super) {
    __extends(MusicMatch, _super);
    function MusicMatch() {
        var _this = _super.apply(this, arguments) || this;
        _this.name = 'MusicMatch';
        return _this;
    }
    MusicMatch.prototype.search = function (title, artist, cb) {
        var _this = this;
        var url = "https://www.musixmatch.com/search/" + encodeURIComponent(artist) + " " + encodeURIComponent(title) + "/tracks";
        this.doReq(url, function (err, res, body) {
            if (err || res.statusCode != 200) {
                return cb('Error response searching music match');
            }
            try {
                var firstUrl = /"track_share_url":"([^"]+)"/.exec(body)[1];
                return _this.getSong(firstUrl, cb);
            }
            catch (e) {
                cb('Music match fail');
            }
        });
    };
    MusicMatch.prototype.parseContent = function (body) {
        var str = body.split('"body":"')[1].replace(/\\n/g, "\n");
        var result = [];
        var len = str.length;
        for (var i = 0; i < len; i++) {
            if (str[i] === '"' && (i === 0 || str[i - 1] !== '\\')) {
                return result.join('');
            }
            else if (str[i] === '"') {
                result.pop();
            }
            result.push(str[i]);
        }
        return result.join('');
    };
    MusicMatch.prototype.getSong = function (url, cb) {
        var _this = this;
        this.doReq(url, function (err, res, body) {
            if (err || res.statusCode != 200) {
                console.log('Err', err);
                console.log('Res', res);
                return cb('Error response getting song from MusicMatch');
            }
            try {
                var lyric = _this.parseContent(body);
                return cb(null, { lyric: lyric, url: url });
            }
            catch (e) {
                cb("Music match fail parsing getSong");
            }
        });
    };
    return MusicMatch;
}(SearchLyrics_1.SearchLyrics));
exports.MusicMatch = MusicMatch;
//# sourceMappingURL=MusicMatch.js.map