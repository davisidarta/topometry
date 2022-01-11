"use strict";
var SearchWikia_1 = require("./plugins/SearchWikia");
var MusicMatch_1 = require("./plugins/MusicMatch");
var Genius_1 = require("./plugins/Genius");
var NormalizeTitles_1 = require("./NormalizeTitles");
var async = require('async');
var plugins = [MusicMatch_1.MusicMatch, SearchWikia_1.SearchWikia, Genius_1.Genius];
var request = require('request').defaults({ timeout: 5000 });
var Searcher = (function () {
    function Searcher() {
        this.normalizer = new NormalizeTitles_1.NormalizeTitles();
        this.loadPlugins();
    }
    Searcher.prototype.loadPlugins = function () {
        this.plugins = plugins.map(function (Plugin) {
            return new Plugin(request);
        });
    };
    Searcher.prototype.search = function (title, artist, cb) {
        var from = { lyric: null, sourceName: '', sourceUrl: '' };
        var normalizedTitle = this.normalizer.normalize(title);
        // run plugins on series
        // if some returns success getting a lyric
        // stop and save the lyric result
        async.detectSeries(this.plugins, function (plugin, callback) {
            plugin.search(normalizedTitle, artist, function (err, result) {
                if (!err) {
                    from.lyric = result.lyric;
                    from.sourceName = plugin.name;
                    from.sourceUrl = result.url;
                }
                callback(null, from);
            });
        }, function (err) {
            cb(err, from);
        });
    };
    return Searcher;
}());
exports.Searcher = Searcher;
//# sourceMappingURL=Searcher.js.map